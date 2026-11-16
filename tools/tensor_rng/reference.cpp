// Qualification reference only. This binary is never linked into the service.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
static void word(std::uint64_t x) {
  for (unsigned i=0;i<8;++i) std::cout.put(static_cast<char>(x>>(8*i)));
}
int main(int argc,char** argv) {
  if(argc!=3)return 1;
  std::mt19937 mt(static_cast<std::uint32_t>(std::stoull(argv[1])));
  auto count=std::stoull(argv[2]);if(count>1000000)return 2;
  std::ostringstream raw;raw<<mt;std::istringstream state(raw.str());
  for(unsigned i=0;i<625;++i){std::uint64_t x=0;if(!(state>>x))return 3;word(x);}
  for(std::uint64_t i=0;i<count;++i)word(mt());
}
