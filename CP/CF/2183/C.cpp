// Not solved

#include <bits/stdc++.h>
using namespace std;
using ll = long long;

ll f(ll x) { return x * (x + 1) / 2; }

int main() {
    int t; cin >> t;
    while (t--) {
        ll n, m, k;
        cin >> n >> m >> k;
        ll L = k - 1, R = n - k, a = 1;
        
        for (ll l = 0; l <= L && f(l) <= m; l++) {
            ll re = m - f(l);
            ll r = (ll)((sqrt(1.0L + 8.0L * re) - 1.0L) / 2.0L);
            while (r > 0 && f(r) > re) r--;
            while (f(r + 1) <= re) r++;
            r = min(r, R);
            a = max(a, 1 + l + r);
        }
        cout << min(n, a) << endl;
    }
}
