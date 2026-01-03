#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
    int t; cin >> t;
    while (t--) {
        int n, m; ll k;
        cin >> n >> m >> k;
        vector<int> a(m);
        for (int& v : a) cin >> v;
        sort(a.rbegin(), a.rend());

        vector<pair<int, ll>> f(n);
        ll s = 0;
        for (int i = 0; i < n; i++) {
            int x, y, z; cin >> x >> y >> z;
            f[i] = {x, z - y}; s += y;
        }
        sort(f.rbegin(), f.rend());

        priority_queue<ll> q;
        int p = n - 1, c = 0;
        for (int j = 0; j < m; j++) {
            while (p >= 0 && f[p].first <= a[j]) q.push(f[p--].second);
            if (!q.empty()) { q.pop(); c++; }
        }

        vector<ll> r;
        while (!q.empty()) { r.push_back(q.top()); q.pop(); }
        for (int i = 0; i <= p; i++) r.push_back(f[i].second);
        sort(r.begin(), r.end());

        ll b = k - s;
        for (ll d : r) if (b >= d) { b -= d; c++; }
        cout << c << endl;
    }
}
