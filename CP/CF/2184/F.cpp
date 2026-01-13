#include <bits/stdc++.h>
using namespace std;

vector<int> g[200100];
int d[200100];

void dfs(int u, int p) {
    vector<int> c;
    for (int v : g[u]) if (v != p) dfs(v, u), c.push_back(v);
    if (c.empty()) { d[u] = 2; return; }
    int r = 2, s = 1;
    for (int x : c) {
        int t = 0;
        for (int i = 0; i < 3; i++) if (s >> i & 1)
            for (int j = 0; j < 3; j++) if (d[x] >> j & 1)
                t |= 1 << (i + j) % 3;
        s = t;
    }
    d[u] = r | s;
}

int main() {
    int t; cin >> t;
    while (t--) {
        int n; cin >> n;
        for (int i = 1; i <= n; i++) g[i].clear();
        for (int i = 1; i < n; i++) {
            int u, v; cin >> u >> v;
            g[u].push_back(v); g[v].push_back(u);
        }
        dfs(1, 0);
        cout << (d[1] & 1 ? "YES" : "NO") <<endl;
    }
}
