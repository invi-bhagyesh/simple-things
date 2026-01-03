#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n, bal = 0, cnt = 0;
        string s;
        cin >> n >> s;
        for (char c : s) cnt += (bal += (c == '1' ? 1 : -1)) >= 0;
        cout << cnt <<endl;
    }
}
