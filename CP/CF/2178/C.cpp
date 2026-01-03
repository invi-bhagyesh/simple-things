#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main(){
    int t;
    cin >> t;
    while(t--){
        int n;
        cin >> n;
        vector<ll> a(n+1), pref(n+1, 0);

        for(int i = 1; i <= n; i++){
            cin >> a[i];
            pref[i] = pref[i-1] + a[i];
        }

        ll total = pref[n];
        ll ans = LLONG_MIN;

        for(int k = 1; k <= n; k++){
            ll left_sum = pref[k-1];
            ll right_sum = total - pref[k];
            ll cur = left_sum - right_sum;
            ans = max(ans, cur);
        }

        cout << ans <<endl;
    }
    return 0;
}
