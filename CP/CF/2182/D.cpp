#include <bits/stdc++.h>
using namespace std;
using ll = long long;

const ll mod = 998244353;
ll f[55], inv[55];

ll pw(ll a, ll b) {
    ll r = 1;
    for (; b; b >>= 1, a = a * a % mod)
        if (b & 1) r = r * a % mod;
    return r;
}

ll c(int n, int k) {
    if (k < 0 || k > n) return 0;
    return f[n] * inv[k] % mod * inv[n - k] % mod;
}

int main() {
    f[0] = 1;
    for (int i = 1; i <= 54; i++) f[i] = f[i - 1] * i % mod;
    inv[54] = pw(f[54], mod - 2);
    for (int i = 53; i >= 0; i--) inv[i] = inv[i + 1] * (i + 1) % mod;

    int t; cin >> t;
    while (t--) {
        int n; cin >> n;
        vector<ll> a(n + 1);
        ll s = 0;
        for (int i = 0; i <= n; i++) { cin >> a[i]; s += a[i]; }

        ll q = s / n, r = s % n, base = 0;
        int sm = 0;
        for (int i = 1; i <= n; i++) {
            base += max(0ll, q - a[i]);
            sm += a[i] < q + 1;
        }

        ll th = a[0] - base;
        if (th < 0) { cout << 0 << endl; continue; }

        ll ans = 0;
        for (int k = max(0, (int)r - (n - sm)); k <= min({th, (ll)r, (ll)sm}); k++)
            ans = (ans + c(sm, k) * c(n - sm, r - k)) % mod;

        cout << ans * f[r] % mod * f[n - r] % mod << endl;
    }
}
