#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n; cin >> n;
    vector<ll> a(n), b(n), c(n);
    for (auto& x : a) cin >> x;
    for (auto& x : b) cin >> x;
    for (auto& x : c) cin >> x;

    vector<ll> pa(n+1), pb(n+1), sc(n+1);
    for (int i = 0; i < n; i++) pa[i+1] = pa[i] + a[i], pb[i+1] = pb[i] + b[i];
    for (int i = n-1; i >= 0; i--) sc[i] = sc[i+1] + c[i];

    ll mx = LLONG_MIN, ans = LLONG_MIN;
    for (int y = 1; y < n-1; y++) {
        mx = max(mx, pa[y] - pb[y]);
        ans = max(ans, mx + pb[y+1] + sc[y+1]);
    }
    cout << ans << endl;
}
