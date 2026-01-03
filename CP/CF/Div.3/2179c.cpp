#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int tc; cin>>tc;
    while(tc--){
        int n; cin>>n;
        ll arr[n], m = 1e18;
        for(int i=0;i<n;i++){ cin>>arr[i]; if(arr[i]<m) m=arr[i]; }
        
        ll ans = m;
        for(int i=0;i<n;i++){
            if(arr[i]==m) continue;
            ll dif = arr[i]-m;
            if(dif <= m) continue;
            ll best = dif;
            for(ll d=1;d*d<=dif;d++){
                if(dif%d==0){
                    if(d>m && d<best) best=d;
                    if(dif/d>m && dif/d<best) best=dif/d;
                }
            }
            if(best<ans) ans=best;
        }
        cout<<ans<<endl;
    }
}
