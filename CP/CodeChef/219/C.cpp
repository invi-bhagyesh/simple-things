#include <bits/stdc++.h>
using ll = long long;
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        vector<ll> a(n);
        for (int i = 0; i < n; i++) cin >> a[i];
        
        ll mn = a[0], mx = 0, sum = 0, pw = 1;
        for (int i = 1; i < n; i++) sum += a[i];
        mn += 2 * sum;
        for (int i = 0; i < n; i++) { mx += a[i] * pw; pw *= 2; }
        cout << mn << " " << mx << endl;
    }
}
