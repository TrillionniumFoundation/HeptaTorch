// Independent qualification reference only; never linked into HIA production.
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

static void word(std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    std::cout.put(static_cast<char>((value >> (8 * i)) & 255));
}
static void state(const std::mt19937& engine) {
  std::ostringstream encoded;
  encoded << engine;
  std::istringstream input(encoded.str());
  for (unsigned i = 0; i < 625; ++i) {
    std::uint64_t value = 0;
    if (!(input >> value)) std::exit(2);
    word(value);
  }
  std::string extra;
  if (input >> extra) std::exit(3);
}
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  const auto seed = std::stoull(argv[1]);
  const auto count = std::stoull(argv[2]);
  if (count > 100000) return 4;
  std::mt19937 engine(static_cast<std::uint32_t>(seed));
  state(engine);
  for (std::uint64_t i = 0; i < count; ++i) word(engine());
  state(engine);
}
