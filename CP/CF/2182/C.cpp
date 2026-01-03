#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
    int t; cin >> t;
    while (t--) {
        int n; cin >> n;
        vector<int> a(n), b(n), c(n);
        for (int& x : a) cin >> x;
        for (int& x : b) cin >> x;
        for (int& x : c) cin >> x;
        
        auto cnt = [&](vector<int>& u, vector<int>& v) {
            int res = 0;
            for (int d = 0; d < n; d++) {
                bool ok = true;
                for (int x = 0; x < n && ok; x++)
                    ok = u[x] < v[(x + d) % n];
                res += ok;
            }
            return res;
        };
        
        cout << (ll)n * cnt(a, b) * cnt(b, c) << endl;
    }
}
