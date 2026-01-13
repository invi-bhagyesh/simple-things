#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int t; cin >> t;
    while (t--) {
        ll n, k; cin >> n >> k;
        int d = 63 - __builtin_clzll(n);
        ll ans = (d >= k);
        for (int l = 1; l <= d; l++) {
            ll th = k - l + 1;
            if (th <= 0) ans += 1LL << (l - 1);
            else for (int x = th; x < l; x++) {
                ll c = 1;
                for (int i = 1; i <= x; i++) c = c * (l - i) / i;
                ans += c;
            }
        }
        cout << ans << endl;
    }
}
