#include <bits/stdc++.h>
using namespace std;

int main() {
    int t; cin >> t;
    while (t--) {
        int n; cin >> n;
        vector<vector<int>> g(n + 1);
        for (int i = 1; i < n; i++) {
            int u, v; cin >> u >> v;
            g[u].push_back(v); g[v].push_back(u);
        }
        vector<int> d(n + 1, -1), cnt(n + 1);
        queue<int> q; q.push(1); d[1] = 0;
        int a = 1, md = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            cnt[d[u]]++; md = max(md, cnt[d[u]]);
            int c = 0;
            for (int v : g[u]) if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); c++; }
            a = max(a, c + 1);
        }
        cout << max(a, md) << endl;
    }
}
