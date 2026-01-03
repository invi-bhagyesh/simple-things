#include <bits/stdc++.h>
using namespace std;

int main() {
    long long t;
    cin >> t;

    while (t--) {
        long long n;
        cin >> n;

        vector<long long> a(n);
        for (long long &t: a){cin>>t;}

        long long sum = 0;
        for (long long i = 0; i + 1 < n; i++) {
            sum += abs(a[i] - a[i + 1]);
        }
        long long ans = sum;
        ans = min(ans, sum - llabs(a[0] - a[1]));
        ans = min(ans, sum - llabs(a[n - 2] - a[n - 1]));

        for (long long k = 1; k + 1 < n; k++) {
            long long one = sum - llabs(a[k - 1] - a[k]) - llabs(a[k] - a[k + 1]) + llabs(a[k - 1] - a[k + 1]);
            ans = min(ans, one);
        }
        cout<<ans<<endl;
    }
}
