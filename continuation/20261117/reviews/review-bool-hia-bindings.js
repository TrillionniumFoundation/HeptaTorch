function inspectHiaBindings(source) {
  const lines=source.trimEnd().split("\n");
  const head=/^fn ([A-Za-z_]\w*)\(([^)]*)\) -> \(([^)]*)\).* entry\((\w+)\)$/.exec(lines[0]);
  if(!head) throw Error("unrecognized function header");
  const parseParams=s=>s?s.split(",").map(x=>{const m=/^(v\d+):(\w+)$/.exec(x);if(!m)throw Error("bad parameter "+x);return [m[1],m[2]];}):[];
  const params=parseParams(head[2]), returns=head[3]?head[3].split(","):[], blocks=new Map(), declarations=new Set(), errors=[], edges=[];
  let block=null;
  for(const line of lines.slice(1)){
    const h=/^block (\w+)\(([^)]*)\)$/.exec(line);
    if(h){if(block&&!block.terminal) errors.push("missing terminal "+block.name);if(blocks.has(h[1]))errors.push("duplicate block "+h[1]);block={name:h[1],params:parseParams(h[2]),values:new Map(),terminal:false};blocks.set(h[1],block);for(const [v,t]of block.params){if(declarations.has(v))errors.push("duplicate value "+v);declarations.add(v);block.values.set(v,t);}continue;}
    if(line==="end") {if(block&&!block.terminal)errors.push("missing terminal "+block.name);continue;}
    if(!block){errors.push("statement before block");continue;}
    if(block.terminal)errors.push("statement after terminal "+block.name);
    const use=v=>{if(!block.values.has(v))errors.push("undefined "+v+" in "+block.name);return block.values.get(v);};
    const ins=/^  (v\d+):(\w+) = ([\w.]+)\(([^)]*)\)(?: .*)?$/.exec(line);
    if(ins){const args=ins[4]?ins[4].split(","):[];for(const v of args)use(v);if(declarations.has(ins[1]))errors.push("duplicate value "+ins[1]);declarations.add(ins[1]);block.values.set(ins[1],ins[2]);continue;}
    const discarded=/^  _ = ([\w.]+)\(([^)]*)\)(?: .*)?$/.exec(line);
    if(discarded){for(const v of (discarded[2]?discarded[2].split(","):[]))use(v);continue;}
    const branch=/^  branch (v\d+) (\w+)\(([^)]*)\) (\w+)\(([^)]*)\)$/.exec(line);
    const jump=/^  jump (\w+)\(([^)]*)\)$/.exec(line);
    const ret=/^  ret\(([^)]*)\)$/.exec(line);
    const edge=(target,s)=>{const args=s?s.split(","):[];edges.push({from:block.name,target,types:args.map(use)});};
    if(branch){if(use(branch[1])!=="bool")errors.push("non-bool branch "+block.name);edge(branch[2],branch[3]);edge(branch[4],branch[5]);block.terminal=true;}
    else if(jump){edge(jump[1],jump[2]);block.terminal=true;}
    else if(ret){const types=(ret[1]?ret[1].split(","):[]).map(use);if(JSON.stringify(types)!==JSON.stringify(returns))errors.push("return type/arity "+block.name);block.terminal=true;}
    else errors.push("unrecognized statement "+line);
  }
  const entry=blocks.get(head[4]);if(!entry||JSON.stringify(entry.params)!==JSON.stringify(params))errors.push("entry parameter mismatch");
  for(const e of edges){const target=blocks.get(e.target);if(!target)errors.push("unknown target "+e.target);else if(JSON.stringify(e.types)!==JSON.stringify(target.params.map(x=>x[1])))errors.push("edge type/arity "+e.from+" -> "+e.target);}
  return {name:head[1],blocks:blocks.size,declared_values:declarations.size,control_flow_edges:edges.length,errors};
}
