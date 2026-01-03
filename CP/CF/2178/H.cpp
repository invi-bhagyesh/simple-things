#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INF = (ll)4e18;

int main() {
    int t;
    cin >> t;

    while (t--) {
        int a, b, c, m, k;
        cin >> a >> b >> c >> m >> k;

        int start = (a + b + c) % m;

        if (start == 0) {
            cout << 0 << '\n';
            continue;
        }

        vector<ll> dist(m, INF);
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;

        dist[start] = 0;
        pq.push({0, start});

        vector<int> vals = {a, b, c};

        while (!pq.empty()) {
            auto [cost, u] = pq.top();
            pq.pop();

            if (cost > dist[u]) continue;
            if (u == 0) break;

            for (int x : vals) {
                int v = (u + x) % m;
                ll nc = cost + x;
                if (nc < dist[v]) {
                    dist[v] = nc;
                    pq.push({nc, v});
                }
            }

            for (int x : vals) {
                ll add = x;
                ll dupCost = k;
                while (add < m) {
                    int v = (u + add) % m;
                    ll nc = cost + dupCost;
                    if (nc < dist[v]) {
                        dist[v] = nc;
                        pq.push({nc, v});
                    }
                    add <<= 1;
                    dupCost += k;
                }
            }
        }

        cout << dist[0] << endl;
    }

    return 0;
}
