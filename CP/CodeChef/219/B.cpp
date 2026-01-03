#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int a, b;
        cin >> a >> b;
        if (4 * b > 9 * a) {
            cout << "Small" << endl;
        } else if (4 * b < 9 * a) {
            cout << "Large" << endl;
        } else {
            cout << "Equal" << endl;
        }
    }
}
