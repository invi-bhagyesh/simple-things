#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int t;
    cin >> t;
    while (t--) {
        ll n, k;
        cin >> n >> k;
        if (k > n) {
            cout << -1 << endl;
            continue;
        }
        if (k == n) {
            cout << 0 << endl;
            continue;
        }
        int d = 1;
        while ((1LL << d) * (k - 1) + 1 <= n && n > (1LL << d) * (k + 1) - 1)
            d++;
        cout << ((1LL << d) * (k - 1) + 1 > n ? -1 : d) << endl;
    }
}
