#include<iostream>
using namespace std;

void matrix_init(double (*M)[4]) {
    for (int i = 0; i < 4; i++) {
        for (int t = 0; t < 4; t++) {
            cin >> M[i][t];
        }
    }
}

void matrix_multi(double (*M1)[4], double (*M2)[4], double (*R)[4]) {
    for (int i = 0; i < 4; i++) {
        for (int t = 0; t < 4; t++) {
            for (int j = 0; j < 4; j++) {
                R[i][t] += M1[i][j] * M2[j][t];
            }
        }
    }
}

void print_matrix(double (*M)[4]) {
    for (int i = 0; i < 4; i++) {
        for (int t = 0; t < 4; t++) {
            cout << M[i][t] << " ";
        }
        cout << endl;
    }
}

int main() {
    double M1[4][4], M2[4][4];

    matrix_init(M1);
    matrix_init(M2);

    double R[4][4] = {0};

    matrix_multi(M1, M2, R);

    print_matrix(R);
    
    return 0;
}