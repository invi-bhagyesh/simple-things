#include <bits/stdc++.h>
using namespace std;

int main() {
    int t; 
    cin >> t;
    while (t--) {
        string s; 
        cin >> s;

        int y = count(s.begin(), s.end(), 'Y');
        int n = s.size() - y;

        cout << (n >= y - 1 ? "YES" : "NO") << endl;
    }
}
