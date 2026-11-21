"""Exact committed FE factory/promotion/result ownership for the pinned773 path.

Derived from scalar_previous_typed_descriptor (672-byte clear, field writes),
half_mul_promote, half_addsub_tensor, descriptor_page, alloc/so_slot_new,
so_extent_allocate, finish_command, graph_collect_since,
retained_row_release_original, so_version_release, so_storage_release and
so_slot_drop. This inspects actual endpoint snapshots. It does not construct
or claim an observed intermediate owner, and it computes no numerical output.

The two original87 FE fixture have sufficient existing slot/pool capacity.
That scope is checked explicitly; a growing owner needs its own reviewed rule.
"""

SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
IMAGE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_fe_promotion_tombstone(owner, before, after, factory, result):
    """Check all84 words and all eight owner tables from source semantics."""
    require(owner.SOURCE == SOURCE and owner.IMAGE == IMAGE and owner.DESCRIPTOR == 672,
            'exact reviewed773 source, native image and descriptor width')
    require(before['header']['slots'] == after['header']['slots']
            and before['header']['pool'] == after['header']['pool'],
            'original FE endpoint capacity unchanged; no unproved owner growth')
    first = before['header']['next']
    birth = before['header']['incarnation']
    indices = (first, first + 1, first + 2)
    require(factory == first and result == first + 2
            and after['header']['next'] == first + 3
            and after['header']['incarnation'] == birth + 3,
            'exact factory, temporary promotion, result descriptor order')
    nodes = [after['nodes'][str(index)] for index in indices]
    dtype = nodes[0][72]
    require(dtype == 4, 'exact original public F64 factory dtype')
    old_pages = {page['page']: page for page in before['pages']}
    new_pages = {page['page']: page for page in after['pages']}
    required_pages = {index >> 8 for index in indices}
    require(set(new_pages) == set(old_pages) | required_pages
            and all(new_pages[page] == item for page, item in old_pages.items()),
            'only descriptor pages reached by the three actual new indices')

    # so_slot_new selects the first nonlive slot whose generation can advance.
    # The three T/V allocations stay live until the second command finishes.
    # No synthetic owner is used: these are arithmetic identities over before.
    for kind in map(str, range(1, 9)):
        require(len(before['tables'][kind]) == len(after['tables'][kind])
                == before['header']['slots'], 'complete unchanged slot capacity')
    available = {}
    for kind in ('1', '2', '3'):
        available[kind] = [slot for slot, row in enumerate(before['tables'][kind])
                           if row[1] == 0 and row[0] < 0xffffffff]
    handles = {kind: [] for kind in available}
    for kind in ('1', '3'):
        require(len(available[kind]) >= 3, 'three existing reusable T/V slots')
        for slot in available[kind][:3]:
            generation = before['tables'][kind][slot][0] + 1
            handles[kind].append((generation << 32) | (int(kind) << 24) | (slot + 1))

    # descriptor_page may allocate one internal page immediately before a
    # descriptor. Each descriptor then allocates one data Storage:16,16,16 bytes.
    # alloc uses16-byte first-fit extents; no payload is freed until promotion
    # cleanup. Its exact retained relative address follows from that plan.
    extents = [list(item) for item in before['free_extents']]
    events = []
    page_seen = set(old_pages)
    for ordinal, index in enumerate(indices):
        page = index >> 8
        if page not in page_seen:
            events.append(dict(role='page', page=page, requested=672 * 256))
            page_seen.add(page)
        events.append(dict(role='data', ordinal=ordinal, requested=16))
    require(len(available['2']) >= len(events), 'all required existing Storage slots')
    data_events = {}
    # alloc also obtains a Hold by the same first-free generation rule.
    # A page retains its Hold; each typed data allocation immediately drops
    # its Hold, so a slot can be reused and incremented again by the next alloc.
    # These rows are algebraic expectations, not fabricated observed states.
    expected_holds = [list(row) for row in before['tables']['6']]
    hold_events = []
    for order, event in enumerate(events):
        size = (event['requested'] + 15) & ~15
        fits = [index for index, (_, length) in enumerate(extents) if length >= size]
        require(fits, 'every exact allocation fits the actual before free extents')
        fit = fits[0]; offset, extent = extents[fit]
        if extent == size:
            del extents[fit]
        else:
            extents[fit] = [offset + size, extent - size]
        slot = available['2'][order]
        generation = before['tables']['2'][slot][0] + 1
        handle = (generation << 32) | (2 << 24) | (slot + 1)
        event.update(offset=offset, extent=size, slot=slot, generation=generation, handle=handle)
        handles['2'].append(handle)
        hslots = [slot for slot, row in enumerate(expected_holds)
                  if row[1] == 0 and row[0] < 0xffffffff]
        require(hslots, 'each exact allocation has an existing reusable Hold slot')
        hslot = hslots[0]
        hgeneration = expected_holds[hslot][0] + 1
        hhandle = (hgeneration << 32) | (6 << 24) | (hslot + 1)
        expected_holds[hslot] = [hgeneration, 1, handle, 1, 0, 0, 0, 0]
        hold_events.append(dict(role=event['role'], slot=hslot,
                                generation=hgeneration, storage_handle=handle,
                                retained=event['role'] == 'page'))
        if event['role'] == 'data':
            data_events[event['ordinal']] = event
            expected_holds[hslot] = [hgeneration] + [0] * 7
        else:
            page = new_pages[event['page']]
            hold = owner._live_row(after, 6, page['hold'], 'new exact descriptor-page Hold')
            require(page['hold'] == hhandle and hold == expected_holds[hslot]
                    and page['address'] == owner.TAG + offset,
                    'actual new descriptor-page Storage follows first-free/first-fit allocation')

    # so_storage_release returns exactly the aligned promotion extent, and
    # so_extent_release orders/coalesces adjacent free extents canonically.
    temporary = data_events[1]
    released = sorted(extents + [[temporary['offset'], temporary['extent']]])
    coalesced = []
    for offset, size in released:
        if coalesced:
            end = coalesced[-1][0] + coalesced[-1][1]
            require(end <= offset, 'released promotion extent has no live/free overlap')
            if end == offset:
                coalesced[-1][1] += size
                continue
        coalesced.append([offset, size])
    require(after['free_extents'] == coalesced,
            'exact final free extents after three allocations and promotion reclamation')

    expected_tables = {kind: [list(row) for row in before['tables'][kind]]
                       for kind in map(str, range(1, 9))}
    expected_tables['6'] = expected_holds
    for event in events:
        retired = event['role'] == 'data' and event['ordinal'] == 1
        expected_tables['2'][event['slot']] = ([event['generation']] + [0] * 7 if retired else
            [event['generation'], 1, event['offset'], event['requested'], event['extent'], 1, 1, 0])
    retired_rows = {}
    for ordinal, (index, node) in enumerate(zip(indices, nodes)):
        data = data_events[ordinal]
        tensor, storage, version = handles['1'][ordinal], data['handle'], handles['3'][ordinal]
        expected = [0] * 84
        expected[0], expected[1], expected[2] = 1, 2, owner.TAG + data['offset']
        expected[8], expected[12], expected[16] = index, 2, 1
        expected[24], expected[27], expected[28] = tensor, storage, version
        expected[40], expected[72] = birth + ordinal + 1, 4
        if ordinal != 1:
            expected[25] = expected[30] = expected[38] = 1
        require(node == expected, 'all84 canonical source-derived descriptor words: ordinal ' + str(ordinal))
        tslot, vslot = (tensor & 0xffffff) - 1, (version & 0xffffff) - 1
        tg, vg = tensor >> 32, version >> 32
        if ordinal == 1:
            expected_tables['1'][tslot] = [tg] + [0] * 7
            expected_tables['3'][vslot] = [vg] + [0] * 7
            retired_rows = {'1': [tg] + [0] * 7, '2': [storage >> 32] + [0] * 7,
                            '3': [vg] + [0] * 7}
        else:
            expected_tables['1'][tslot] = [tg, 1, storage, version, 0, data['requested'], index, 0]
            expected_tables['3'][vslot] = [vg, 1, 0, 1, 0, 0, 0, 0]
    for kind, expected in expected_tables.items():
        require(after['tables'][kind] == expected,
                'every original and allocated/retired owner table word follows source: kind ' + kind)
    return dict(schema='heptatorch.exact-F64-FE-promotion-tombstone.v1',
                descriptor_indices=list(indices), promotion_index=first + 1,
                all_three_complete84word_descriptors_checked=True,
                promotion_all84words_source_derived=True,
                exact_retired_Tensor_Storage_Version_rows=retired_rows,
                exact_slot_first_free_generations=True, exact_storage_first_fit_offsets=True,
                exact_Hold_allocation_retirement_events=hold_events,
                complete_all_eight_owner_tables_checked=True,
                Weak_Saved_Generator_Optimizer_tables_unchanged_including_dead_rows=True,
                promotion_data_extent_returned_to_actual_free_list=True,
                descriptor_pages_only_for_allocated_indices=True,
                observed_capacity_unchanged=True, intermediate_owner_snapshots_inferred=False,
                numerical_expected_values_computed=False)
