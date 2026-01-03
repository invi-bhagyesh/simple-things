#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
    int t; cin >> t;
    while (t--) {
        ll a, b;
        cin >> a >> b;
        int ans = 0;
        for (int n = 1; n <= 40; n++) {
            ll o = 0, e = 0, s = 1;
            for (int i = 1; i <= n; i++, s *= 2)
                (i & 1 ? o : e) += s;
            if ((o <= a && e <= b) || (o <= b && e <= a))
                ans = n;
        }
        cout << ans << endl;
    }
}
