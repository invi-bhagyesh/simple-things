#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    if (!(cin >> t)) return 0;
    while (t--) {
        int n; cin >> n;
        vector<vector<int>> g(n + 1);
        for (int i = 1; i < n; i++) {
            int u, v; cin >> u >> v;
            g[u].push_back(v); g[v].push_back(u);
        }

        vector<int> d(n + 1, -1), p(n + 1), o;
        vector<int> x(n + 1, 0);
        queue<int> q;
        q.push(1);
        d[1] = 0;
        p[1] = 0;

        int k = 0;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            o.push_back(u);
            x[d[u]]++;
            k = max(k, x[d[u]]);

            int y = 0;
            for (int v : g[u]) {
                if (d[v] == -1) {
                    d[v] = d[u] + 1;
                    p[v] = u;
                    q.push(v);
                    y++;
                }
            }
            k = max(k, y + 1);
        }

        vector<int> c(n + 1);
        vector<vector<int>> b(n + 1);
        for (int u : o) b[d[u]].push_back(u);

        c[1] = 1;

        for (int i = 1; i <= n; i++) {
            if (b[i].empty()) break;

            auto& v = b[i];
            sort(v.begin(), v.end(), [&](int x, int y) {
                return c[p[x]] < c[p[y]];
            });

            int m = v.size();
            vector<int> z;
            z.reserve(m);
            for (int j = 0; j < m; j++) {
                int f = c[p[v[j]]];
                int val = (f - 1 - j) % k;
                if (val < 0) val += k;
                z.push_back(val);
            }
            sort(z.begin(), z.end());

            int s = -1;
            int l = -1;
            for (int val : z) {
                if (val > l + 1) {
                    s = l + 1;
                    break;
                }
                l = val;
            }
            if (s == -1) s = l + 1;

            for (int j = 0; j < m; j++) {
                c[v[j]] = ((j + s) % k) + 1;
            }
        }

        cout << k <<endl;
        vector<vector<int>> r(k + 1);
        for (int u = 1; u <= n; u++) r[c[u]].push_back(u);
        for (int i = 1; i <= k; i++) {
            cout << r[i].size();
            for (int u : r[i]) cout << " " << u;
            cout <<endl;
        }
    }
}
