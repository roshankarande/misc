#include <iostream>
using namespace std;
#include <tuple>

// Return multiple values

std::tuple<int, int> divide(int dividend, int divisor)
{
    return {dividend / divisor, dividend % divisor};
}

int main()
{
    using namespace std;

    auto [quotient, remainder] = divide(14, 3);

    cout << quotient << ',' << remainder << endl;
}

// Available from C++17
// g++ -std=c++17 main.cpp