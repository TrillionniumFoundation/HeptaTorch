// Independent qualification only: standard-library MT19937, imported words.
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

static std::uint64_t read_word() {
  std::uint64_t value = 0;
  for (unsigned i=0;i<8;++i) {
    int c=std::cin.get(); if(c<0) std::exit(2);
    value |= static_cast<std::uint64_t>(c) << (8*i);
  }
  return value;
}
static void word(std::uint64_t value) {
  for (unsigned i=0;i<8;++i) std::cout.put(static_cast<char>((value>>(8*i))&255));
}
int main() {
  std::ostringstream text;
  for(unsigned i=0;i<624;++i) text << read_word() << ' ';
  auto index=read_word(), count=read_word();
  if(index>624 || count>1000000) return 3;
  text << index;
  std::mt19937 engine;
  std::istringstream encoded(text.str());
  if(!(encoded>>engine)) return 4;
  if(std::cin.get()!=std::char_traits<char>::eof()) return 5;
  for(std::uint64_t i=0;i<count;++i) word(engine());
  std::ostringstream result;result<<engine;
  std::istringstream state(result.str());
  for(unsigned i=0;i<625;++i) { std::uint64_t v=0;if(!(state>>v))return 6;word(v); }
  return 0;
}
