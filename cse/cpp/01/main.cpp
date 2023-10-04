#include <iostream>

int add(int x, int y)
{
    return x + y;
}

int &mul(int x, int y)
{
    static int z = x * y;
    return z;
}

void set_value(int &x) {}

// accepts both rvalue and lvalues
void PrintName(const std::string &name)
{
    std::cout << name << std::endl;
}

// only accets lvalues
void DisplayName(std::string &name)
{
    std::cout << name << std::endl;
}

// only accepts rvalues
void showName(std::string &&name)
{
    std::cout << name << std::endl;
}

int main(int argc, char const *argv[])
{
    int x = 10; // this is value because x is lvalue and 10 is rvalue
    // 10 = x;   // this is invalid because 10 ("literal") is not rvalue
    int y = 20;

    int a = add(2, 3); // this is value as add(2, 3) returns a temporary value which is a rvalue
    // add(2,3) = 10; // this is in valid as LHS is not lvalue

    int b0 = add(2, 3); // here calling add by passing rvalues
    int b1 = add(x, y); // here calling add by passing lvalues

    int c = mul(2, 3); // this is valid

    mul(2, 3) = 10; // this is valid as mul(2,3) returns a reference which is a lvalue i.e. it is lvalue reference

    // set_value(10); // this won't work as function set_value(int &x) wants a lvalue reference
    set_value(x); // this is valid as we are passing a lvalue which can be set to a lvalue reference

    // int& m = 10; // this is not valid
    const int &m = 10; // this is valid because of const ... this is kind of an exception that compiler gives us for convenience.

    std::string first = "Roshan"; // first is lvalue and "Roshan" is value
    std::string last = "K";

    std::string fullname = first + last; // fullname is lvalue and first + last is rvalue as a temporary var gets created hence a rvalue

    DisplayName(fullname); // works with lvalue
    // DisplayName(first + last); // does not work with rvalue

    // showName(fullname); // does not work with lvalue
    showName(first + last); // only works with rvalue

    // Hence you will see a lot of const in function defintions in C++
    PrintName(fullname);     // works with lvalue
    PrintName(first + last); // works with rvalue

    std::cout << "Done!" << std::endl;

    return 0;
}

// Note!!
// We can do function overloading using lvalue and rvalue parameter types... as they would different signatures.
