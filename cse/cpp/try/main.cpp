#include <iostream>
using namespace std;

int main(int argc, char const *argv[])
{
    int n;
    cin >> n;

    int arr[n]{0};

    // for (int i = 0; i < n; i++)
    // {
    //     cout<<arr[i];
    // }

    for (auto &&i : arr)
    {
        cout << i << " ";
    }

    return 0;
}
