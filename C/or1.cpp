#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

static unsigned long long cnt = 0;

struct Node {
    int idx;
    vector<Node*> neighbors;
    bool is_visit;
    bool is_goal;

    Node(int idx) : idx(idx), is_visit(false), is_goal(false) { }

    void Count() {
        if (is_goal) { cnt++;  return; }
        is_visit = true;
        for (vector<Node*>::iterator p = neighbors.begin(); p != neighbors.end(); ++p)
            if (!(*p)->is_visit) (*p)->Count();
        is_visit = false;
    }
};

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " N" << endl;
        return 1;
    }

    int N;
    vector<Node*> nodes;
    stringstream ss;

    ss << argv[1];  ss >> N;
    for (int i = 0; i < N * N; i++) nodes.push_back(new Node(i));
    nodes[N*N-1]->is_goal = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int x = i * N + j;
            if (x - N >= 0) nodes[x]->neighbors.push_back(nodes[x-N]);
            if (x + N < N * N) nodes[x]->neighbors.push_back(nodes[x+N]);
            if (x % N != 0) nodes[x]->neighbors.push_back(nodes[x-1]);
            if (x % N != N - 1) nodes[x]->neighbors.push_back(nodes[x+1]);
        }
    }

    nodes[0]->Count();
    cout << "Count all paths from (0, 0) to ("
         << N - 1 << ", " << N - 1 << ") in the "
         << N << "x" << N << " lattice graph:" << endl;
    cout << "Count: " << cnt << endl;
 
    return 0;
}
