#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    int t; cin >> t;
    while(t--){
        int n, m; cin >> n >> m;
        vector<ll> a(n);
        for(ll &x : a) cin >> x;

        vector<int> id(n);
        iota(id.begin(), id.end(), 0);
        sort(id.begin(), id.end(), [&](int i, int j){ return a[i] < a[j]; });

        if(m == n){
            cout << -1 << '\n';
            continue;
        }
        
        if(m >= 2 && m > n - m){
            cout << -1 << '\n';
            continue;
        }

        vector<pair<int,int>> atk;

        if(m >= 2){
            for(int i = 0; i < m; i++)
                atk.push_back({id[n - m + i] + 1, id[i] + 1});
        } else if(m == 1){
            for(int i = 0; i < n - 1; i++)
                atk.push_back({id[i] + 1, id[i + 1] + 1});
        } else {
            if(n == 2){
                cout << -1 << '\n';
                continue;
            }
            vector<ll> h = a;
            vector<bool> attacked(n, false);
            
            while(true){
                int attacker = -1;
                for(int i = 0; i < n; i++)
                    if(h[id[i]] > 0 && !attacked[id[i]]){ attacker = id[i]; break; }
                if(attacker == -1) break;
                
                int target = -1;
                for(int i = n - 1; i >= 0; i--)
                    if(h[id[i]] > 0 && id[i] != attacker){ target = id[i]; break; }
                if(target == -1) break;
                
                atk.push_back({attacker + 1, target + 1});
                attacked[attacker] = true;
                h[attacker] -= a[target];
                h[target] -= a[attacker];
            }
            
            int alive = 0;
            for(int i = 0; i < n; i++) if(h[i] > 0) alive++;
            if(alive != 0){
                cout << -1 << '\n';
                continue;
            }
        }

        cout << atk.size() << '\n';
        for(auto& [x, y] : atk) cout << x << " " << y << '\n';
    }
}
