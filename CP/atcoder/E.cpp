#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const int LOG = 30;

int main() {

    int n, q;
    cin >> n >> q;

    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        a[i]--; 
    }

    vector<vector<int>> nxt(LOG, vector<int>(n));
    vector<vector<ll>> sm(LOG, vector<ll>(n));

    for (int i = 0; i < n; i++) {
        nxt[0][i] = a[i];
        sm[0][i] = i + 1;
    }

    for (int k = 1; k < LOG; k++) {
        for (int i = 0; i < n; i++) {
            nxt[k][i] = nxt[k-1][ nxt[k-1][i] ];
            sm[k][i] = sm[k-1][i] + sm[k-1][ nxt[k-1][i] ];
        }
    }

    while (q--) {
        ll t;
        int b;
        cin >> t >> b;
        b--;

        ll ans = 0;
        int cur = b;

        for (int k = 0; k < LOG; k++) {
            if (t & (1LL << k)) {
                ans += sm[k][cur];
                cur = nxt[k][cur];
            }
        }

        cout << ans << endl;
    }
}
