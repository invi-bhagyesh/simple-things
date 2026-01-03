#include <bits/stdc++.h>
using namespace std;
using ll = long long;

ll que(int l, int r){
    cout << "? " << l << " " << r << endl;
    cout.flush();
    ll s;
    cin >> s;
    if(s == -1) exit(0);
    return s;
}

int main(){
    int t;
    cin >> t;
    while(t--){
        int n;
        cin >> n;
        ll total = que(1, n);
        ll ans = 1;
        for(int p = 0; p <= 30; p++){
            ll x = 1LL << p;
            if(total > (ll)n * (x - 1))
                ans = x;
        }
        cout << "! " << ans << endl;
        cout.flush();
    }
}
