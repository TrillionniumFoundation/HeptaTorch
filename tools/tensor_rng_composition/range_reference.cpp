// Qualification only: the pinned CPU wrapper's actual binary64 predicate.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
int main() {
  std::uint32_t words[2];
  while (std::cin.read(reinterpret_cast<char*>(words), sizeof words)) {
    float a,b; std::memcpy(&a,&words[0],4); std::memcpy(&b,&words[1],4);
    const double from=static_cast<double>(a),to=static_cast<double>(b);
    const double difference=to-from;
    const std::uint64_t accepted=std::isfinite(a)&&std::isfinite(b)&&from<=to&&
      difference<=std::numeric_limits<float>::max();
    std::uint64_t width; std::memcpy(&width,&difference,8);
    std::cout.write(reinterpret_cast<const char*>(&accepted),8);
    std::cout.write(reinterpret_cast<const char*>(&width),8);
  }
  return std::cin.eof()?0:2;
}
