#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int t;
    cin >> t;
    
    while (t--) {
        ll s, k, m;
        cin >> s >> k >> m;
        
        ll n = m / k;
        ll r = m % k;
        
        ll a;
        if (s < k) {
            a = s;
        } else {
            a = (n % 2 == 0) ? s : k;
        }
        
        cout << max(0LL, a - r) <<endl;
    }
}
