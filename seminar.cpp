//============================================================================
// Name        : seminar.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <sys/time.h>
#include "emmintrin.h"
#include <omp.h>

#define BLOCKFAKI 1000
#define BLOCKFAKJ 1000

typedef double type;
typedef std::vector<type> typeVec;
typedef std::vector<int> intVec;

// template class for a 2D grid
template<typename T> class grid {

private:
	size_t lengthInX;
	size_t lengthInY;
	std::vector<T> data;

public:
	grid() {
		lengthInX = 0;
		lengthInY = 0;
	}
	grid(size_t xDim, size_t yDim) {
		data = std::vector<T>((xDim+1) * yDim, (T) (0.0));
		lengthInX = xDim+1;
		lengthInY = yDim;
	}		// Standart-constructor
	grid(size_t xDim, size_t yDim, T value) {
		data = std::vector<T>((xDim+1) * yDim, value);
		lengthInX = xDim+1;
		lengthInY = yDim;
	}	// initalisation-constructor
	virtual ~grid() {
	}		// Destructor

	size_t lengthX() {
		return lengthInX;
	}	// returns number of elements in x direction
	size_t lengthY() {
		return lengthInX;
	}	// returns number of elements in y direction

	T& operator()(size_t i, size_t j) {
		assert(i < lengthInX-1);
		assert(j < lengthInY);
		return data[j * lengthInX + i];
	}	// return a element at [j*nx+i] so i=x and j=y

};

//___________________________________________________________________________________________________________________


void print_time(std::string text, struct timeval t0, struct timeval t1, int level, std::ostream& file){
    file << text << "\t \t"
            << ((int64_t) (t1.tv_sec - t0.tv_sec) * (int64_t) 1000000
                    + (int64_t) t1.tv_usec - (int64_t) t0.tv_usec) * 1e-3 << std::endl;
}


// calculates the l2-norm of the residual
double residuum(int n, grid<type> &u, grid<type> &f, double h) {
	double residuum = 0.0;
	double sum = 0.0;
	for (int i = 1; i < n - 1; i++) {
		for (int k = 1; k < n - 1; k++) {
			double temp = ((u(k - 1, i) + u(k + 1, i))
					+ (u(k, i - 1) + u(k, i + 1)) - (4 * u(k, i))) / (h * h)
					+ f(k, i);
			sum += temp * temp;
		}
	}
	residuum = sqrt((1.0 / ((n - 1) * (n - 1))) * sum);
	return residuum;
}
/*
// calculates the residual
void residual(int n, grid<type> &u, grid<type> &f, grid<type> &res, double h) {
	for (int i = 1; i < n - 1; i++) {
        for (int k = 1; k < n - 1; k++) {
            if (!(i == n / 2  && k >= n / 2 )) {
			res(k, i) = ((u(k - 1, i) + u(k + 1, i))
					+ (u(k, i - 1) + u(k, i + 1)) - (4 * u(k, i))) / (h * h)
					+ f(k, i);
            }
		}
	}
}

*/
void residual(int n, grid<type> &u, grid<type> &f, grid<type> &res, double h) {
	double h2 = h * h;
	__m128d links, rechts, mitte, oben, unten, rightHandSide, erg1, erg2;
	__m128d const4 = _mm_set_pd(4, 4);
	__m128d meshsize = _mm_load_pd1(&h2);

	#pragma omp parallel for private(links, rechts, mitte, oben, unten, rightHandSide, erg1, erg2)
	for (int i = 1; i < n - 1; i++) {

		res(1, i) = ((u(0, i) + u(2, i)) + (u(1, i - 1) + u(1, i + 1))
				- (4 * u(1, i))) / (h * h) + f(1, i);

		for (int k = 2; k < n - 1; k = k + 2) {

			links = _mm_loadu_pd(&u(k - 1, i));
			rechts = _mm_loadu_pd(&u(k + 1, i));
			erg1 = _mm_add_pd(links, rechts);

			oben = _mm_loadu_pd(&u(k, i + 1));
			unten = _mm_loadu_pd(&u(k, i - 1));
			erg2 = _mm_add_pd(unten, oben);

			erg1 = _mm_add_pd(erg1, erg2);

			mitte = _mm_loadu_pd(&u(k, i));
			mitte = _mm_mul_pd(mitte, const4);
			erg2 = _mm_sub_pd(erg1, mitte);

			erg1 = _mm_div_pd(erg2, meshsize);
			rightHandSide = _mm_loadu_pd(&f(k, i));
			erg1 = _mm_add_pd(erg1, rightHandSide);

			_mm_stream_pd(&res(k, i), erg1); 

		}
	}
}


// restrict a grid (from) with the full weighting stencile to another grid (to). From has size nx[l] and to has size nx[l-1].
void coarsening(int l, grid<type>& from, grid<type>& to, intVec& n) {
#pragma omp parallel for
	for (int i = 1; i < n[l - 1] - 1; i++) {
		for (int j = 1; j < n[l - 1] - 1; j++) {
			if (!(i == n[l - 1] / 2  && j >= n[l - 1] / 2 )) {
				to(j, i) = (from(2 * j - 1, 2 * i + 1)
						+ 2 * from(2 * j, 2 * i + 1)
						+ from(2 * j + 1, 2 * i + 1)
						+ 2 * from(2 * j - 1, 2 * i) + 4 * from(2 * j, 2 * i)
						+ 2 * from(2 * j + 1, 2 * i)
						+ from(2 * j - 1, 2 * i - 1)
						+ 2 * from(2 * j, 2 * i - 1)
						+ from(2 * j + 1, 2 * i - 1)) / 16.0;
			}
		}
	}
}

/*
// restrict a grid (from) with the full weighting stencile to another grid (to). From has size nx[l] and to has size nx[l-1].
void coarsening(int l, grid<type>& from, grid<type>& to, intVec& n) {

  
type* buf = (type*)malloc( 2*sizeof(type) );
__m128d erg;
   
  //#pragma omp parallel for private(reg1, reg2, reg3, reg4, reg5)
	for (int i = 1; i < n[l - 1] - 1; i++) {
	  				to(1, i) = (from(1, 2 * i + 1)
						+ 2 * from(2, 2 * i + 1)
						+ from(3, 2 * i + 1)
						+ 2 * from(1, 2 * i) + 4 * from(2, 2 * i)
						+ 2 * from(3, 2 * i)
						+ from(1, 2 * i - 1)
						+ 2 * from(2, 2 * i - 1)
						+ from(3, 2 * i - 1)) / 16.0;
					
		for (int j = 2; j < n[l - 1] - 1; j= j + 2) {
			if (!(i == n[l - 1] / 2  && j >= n[l - 1] / 2 )) {
			  		buf[0] = (from(2 * j - 1, 2 * i + 1)
						+ 2 * from(2 * j, 2 * i + 1)
						+ from(2 * j + 1, 2 * i + 1)
						+ 2 * from(2 * j - 1, 2 * i) + 4 * from(2 * j, 2 * i)
						+ 2 * from(2 * j + 1, 2 * i)
						+ from(2 * j - 1, 2 * i - 1)
						+ 2 * from(2 * j, 2 * i - 1)
						+ from(2 * j + 1, 2 * i - 1)) / 16.0;
						
					buf[1] = (from(2 * (j+1) - 1, 2 * i + 1)
						+ 2 * from(2 * (j+1), 2 * i + 1)
						+ from(2 * (j+1) + 1, 2 * i + 1)
						+ 2 * from(2 * (j+1) - 1, 2 * i) + 4 * from(2 * (j+1), 2 * i)
						+ 2 * from(2 * (j+1) + 1, 2 * i)
						+ from(2 * (j+1) - 1, 2 * i - 1)
						+ 2 * from(2 * (j+1), 2 * i - 1)
						+ from(2 * (j+1) + 1, 2 * i - 1)) / 16.0;
						
						
					erg = _mm_load_pd (&buf[0]);
					_mm_stream_pd (&to(j, i), erg );
		
			}
		}
	}
}
*/

void interpolation(int l, grid<type>& from, grid<type>& to, intVec& n) {
#pragma omp parallel for
	for (int i = 1; i < n[l] - 1; i++) {
		for (int j = 1; j < n[l] - 1; j++) {
			if (!(i == n[l] / 2  && j >= n[l] / 2)) {
				if (i % 2 == 0 && j % 2 == 0) {
					// wert uebernehmen
					to(j, i) = from(j / 2, i / 2);
				} else if (i % 2 != 0 && j % 2 != 0) {
					// kreuz
					to(j, i) = (type) (from(j / 2, i / 2)
							+ from(j / 2, (i / 2) + 1)
							+ from((j / 2) + 1, i / 2)
							+ from((j / 2) + 1, (i / 2) + 1)) / 4.0;
				} else if (i % 2 == 0 && j % 2 != 0) {
					// vertikal
					to(j, i) = (double) (from(j / 2, i / 2)
							+ from((j / 2) + 1, i / 2)) / 2.0;
				} else {
					// horizontal
					to(j, i) = (double) (from(j / 2, i / 2)
							+ from(j / 2, (i / 2) + 1)) / 2.0;
				}
			}
		}
	}
}

void Red_Black_Gauss(int n, grid<type> &u, grid<type> &f, double h,
		int numIterations) {

	for (int iterations = 0; iterations < numIterations; iterations++) {
	  #pragma omp parallel for
		for (int m = 1; m < n - 1; m++) {
			int q = 1;
			if (m % 2 == 0) {
				q++;
			}
			for (; q < n - 1; q = q + 2) {
				if (!(m == n / 2  && q >= n / 2 )) {
					u(q, m) = (1.0 / 4.0)
							* (h * h * f(q, m)
									+ (u(q - 1, m) + u(q + 1, m) + u(q, m - 1)
											+ u(q, m + 1)));
				}
			}
		}
		#pragma omp parallel for
		for (int m = 1; m < n - 1; m++) {
			int q = 1;
			if (m % 2 != 0) {
				q++;
			}
			for (; q < n - 1; q = q + 2) {
				if (!(m == n / 2  && q >= n / 2 )) {
					u(q, m) = (1.0 / 4.0)
							* (h * h * f(q, m)
									+ (u(q - 1, m) + u(q + 1, m) + u(q, m - 1)
											+ u(q, m + 1)));
				}
			}
		}
	}

}

void Jacobi(int n, grid<type> &u, grid<type> &u_neu, grid<type> &f, double h, int numIterations) {

    for (int iterations = 0; iterations < numIterations; iterations++) {
      #pragma omp parallel for
        for (int m = 1; m < n - 1; ++m) {
            for (int q=1; q < n - 1; ++q) {
                if (!(m == n / 2  && q >= n / 2 )) {
                    u_neu(q, m) = (1.0 / 4.0) * (h * h * f(q, m) + (u(q - 1, m) + u(q + 1, m) + u(q, m - 1) + u(q, m + 1)));
                }
            }
        }

        std::swap(u, u_neu);
    }

}


/*
void Jacobi(int n, grid<type> &u, grid<type> &u_neu, grid<type> &f, double h, int numIterations) {

    __m128d reg1, reg2, reg3, reg4;
    __m128d cost1over4 = _mm_set_pd1(0.25);
    __m128d constH2 = _mm_set_pd1( h*h);

    for (int iterations = 0; iterations < numIterations; iterations++) {
      #pragma omp parallel for private(reg1, reg2, reg3, reg4)
        for (int m = 1; m < n - 1; ++m) {
	    u_neu(1, m) = (1.0 / 4.0) * (h * h * f(1, m) + (u(0, m) + u(2, m) + u(1, m - 1) + u(1, m + 1)));
            for (int q=2; q < n - 1; q=q+2) {
                if (!(m == n / 2  && q >= n / 2 )) {
	
		    reg1 = _mm_loadu_pd( &f(q,m));
		    reg2 = _mm_mul_pd( reg1, constH2);
		    
		    reg3 = _mm_loadu_pd(&u(q-1,m));
		    reg4 = _mm_loadu_pd(&u(q+1,m));
		    reg1 = _mm_add_pd( reg3, reg4 );
		    reg3 = _mm_add_pd( reg1, reg2 );
		    
		    reg1 = _mm_load_pd(&u(q,m-1));
		    reg2 = _mm_load_pd(&u(q,m+1));
		    reg4 = _mm_add_pd( reg1, reg2);
		    
		    reg1 = _mm_add_pd( reg3, reg4 );

		    reg2 = _mm_mul_pd( reg1, cost1over4 );
		    _mm_stream_pd (&u_neu(q, m), reg2 );
                    //u_neu(q, m) = (1.0 / 4.0) * (h * h * f(q, m) + (u(q - 1, m) + u(q + 1, m) + u(q, m - 1) + u(q, m + 1)));
                }
            }
        }

        std::swap(u, u_neu);
    }

}
*/

/*

//Parameter "finest" wird fuer die Neuman-RB (GHOST) benoetigt. Damit wird nur beim feinsten Grid die Neumann-RB im Red-Black-Gauss berechnet
void solveMG(int l, std::vector<grid<type>>& u, std::vector<grid<type>>& f,
		intVec& n, std::vector<type>& h, std::vector<grid<type>>& res, int v1=2, int v2=1, int gamma=1) {

    std::ofstream messung;
    messung.open("Messausgabe.txt");
    struct timeval t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
    gettimeofday(&t0, NULL);

	//Presmoothing
    Red_Black_Gauss(n[l], u[l], f[l], h[l], v1);

    gettimeofday(&t1, NULL);


	// Residuum
	residual(n[l], u[l], f[l], res[l], h[l]);

    gettimeofday(&t2, NULL);

	// restrict residual
	coarsening(l, res[l], f[l - 1], n);

    gettimeofday(&t3, NULL);

	if (l <= 1) {
		// solve
        Red_Black_Gauss(n[l - 1], u[l - 1], f[l - 1], h[l - 1], 1);
        gettimeofday(&t4, NULL);
        print_time("RBGauss solve ", t3, t4, l, messung);

	} else {
		for (int i = 1; i < n[l - 1] - 1; i++) {
			for (int j = 1; j < n[l - 1] - 1; j++) {
				u[l - 1](j, i) = 0.0;
			}
		}
        gettimeofday(&t5, NULL);
		for (int i = 0; i < gamma; i++) {
			solveMG(l - 1, u, f, n, h, res, v1, v2, gamma);
		}


		// interpolation
		grid<type> correction(n[l], n[l], 0.0);
        gettimeofday(&t6, NULL);
		interpolation(l, u[l - 1], correction, n);
        gettimeofday(&t7, NULL);

		// correction
		for (int i = 1; i < n[l] - 1; i++) {
			for (int j = 1; j < n[l] - 1; j++) {
				u[l](j, i) += correction(j, i);
			}
		}
        gettimeofday(&t8, NULL);
	}

    gettimeofday(&t9, NULL);
	//Postsmothing
    Red_Black_Gauss(n[l], u[l], f[l], h[l], v2);
    gettimeofday(&t10, NULL);

    if(l == 10){
        print_time("RBGauss Pre ", t0, t1, l, messung);
        print_time("Residuum ", t1, t2, l, messung);
        print_time("Restrict Residual ", t2, t3, l, messung);
        print_time("set u = 0 ", t3, t5, l, messung);
        print_time("Interpolation ", t6, t7, l, messung);
        print_time("Correction ", t7, t8, l, messung);
        print_time("RBGauss Post ", t9, t10, l, messung);
    }

    messung.close();
}

*/

void solveMG(int l, std::vector<grid<type>>& u, std::vector<grid<type>>& u_neu, std::vector<grid<type>>& f,
             intVec& n, std::vector<type>& h, std::vector<grid<type>>& res, int v1, int v2, int gamma) {
    

    
    //Presmoothing
    //Red_Black_Gauss(n[l], u[l], f[l], h[l], v1);
    Jacobi(n[l], u[l], u_neu[l], f[l], h[l], v1);

    // Residuum
    residual(n[l], u[l], f[l], res[l], h[l]);

    // restrict residual
    coarsening(l, res[l], f[l - 1], n);
    
    if (l <= 1) {
        // solve
        //Red_Black_Gauss(n[l - 1], u[l - 1], f[l - 1], h[l - 1], 1);
        Jacobi(n[l-1], u[l-1], u_neu[l-1], f[l-1], h[l-1], 1);
        
    } else {
        for (int i = 1; i < n[l - 1] - 1; i++) {
            for (int j = 1; j < n[l - 1] - 1; j++) {
                u[l - 1](j, i) = 0.0;
            }
        }
        for (int i = 0; i < gamma; i++) {
            solveMG(l - 1, u, u_neu, f, n, h, res, v1, v2, gamma);
        }
        
        
        // interpolation
        grid<type> correction(n[l], n[l], 0.0);
        interpolation(l, u[l - 1], correction, n);
        
        // correction
	#pragma omp parallel for
        for (int i = 1; i < n[l] - 1; i++) {
            for (int j = 1; j < n[l] - 1; j++) {
                u[l](j, i) += correction(j, i);
            }
        }

    }
    
    //Postsmothing
    //Red_Black_Gauss(n[l], u[l], f[l], h[l], v2);
    Jacobi(n[l], u[l], u_neu[l], f[l], h[l], v2);
    
}


int main(int argc, char **argv) {

    //omp_set_num_threads(4);
    int l;
    int v1 = 2;
    int v2 = 1;
    int gamma = 2;
    int cycle = 4;
    
	// Ueberpruefung, ob Eingabeparamter passen
    if(argc == 2){
        l = atoi(argv[1]);
    }
    else if (argc == 6){
        l = atoi(argv[1]);
        v1 = atoi(argv[2]);
        v2 = atoi(argv[3]);
        gamma = atoi(argv[4]);
        cycle = atoi(argv[5]);
    }else{
        fprintf(stderr, "Usage: ./mgsolve levels\n");
        exit(EXIT_SUCCESS);
    }

	intVec n(l, 0);	// total number of gird points in x and y-direction
	typeVec h(l, 0.0);// mesh size of each levels, where h[0] is the mesh size of the coarsesed grid
    std::vector<grid<type>> f(l);	// vector of grids for the rig1ht hand side
	std::vector<grid<type>> res(l);	// vector of grids for the residuums
    std::vector<grid<type>> u(l);	// vector of grids for the approximation u
    std::vector<grid<type>> u_neu(l);	// vector of grids for the approximation u_neu

	// initialisation -------------------------------------------------------------------------------------------

	// vectors for mesh size and number of grid points
	for (int i = l - 1; i >= 0; i--) {
		n[i] = (int) (pow(2, i + 1) + 1);
		h[i] = 2.0 / (n[i] - 1);
	}

	// Anlegen von f:
	for (int i = l - 1; i >= 0; i--) {
		f[i] = grid<type>(n[i], n[i], 0.0);
	}

	// Speicher fuer das residual allokieren
	for (int i = l - 1; i >= 0; i--) {
		res[i] = grid<type>(n[i], n[i], 0.0);
	}

	// Initialisierung des Gitters
	for (int i = l - 1; i >= 0; i--) {
		u[i] = grid<type>(n[i], n[i], 0.0);
	}

    // Initialisierung des Gitters
    for (int i = l - 1; i >= 0; i--) {
      u_neu[i] = grid<type>(n[i], n[i], 0.0);
    }

	// Horizontale und vertikale Randpunkte setzen
	u[l - 1](n[l - 1] - 1, n[l - 1] / 2) = 0.0;	// Rechts
	u[l - 1](0, n[l - 1] / 2) = 1.0;	// Links
	u[l - 1](n[l - 1] / 2, n[l - 1] - 1) = sin(0.25 * M_PI);	// Oben
	u[l - 1](n[l - 1] / 2, 0) = sin(0.75 * M_PI);	// Unten

	// Randpunkte setzen, die von 0 bis 1 laufen
	#pragma omp parallel for
	for (int i = n[l - 1] / 2 + 1; i < n[l - 1]; i++) {
		double y = -1.0 + i * h[l - 1];

		// rechte Seite oben
		u[l - 1](n[l - 1] - 1, i) = sqrt(sqrt(y * y + 1)) * sin(0.5 * atan(y));

		// obere Seite rechts
		u[l - 1](i, n[l - 1] - 1) = sqrt(sqrt(y * y + 1))
				* sin(0.5 * atan(1.0 / y));

		// untere Seite rechts
		u[l - 1](i, 0) = sqrt(sqrt(y * y + 1))
				* sin(0.5 * (atan(y) + ((3.0 / 2.0) * M_PI)));

		// linke Seite oben
		u[l - 1](0, i) = sqrt(sqrt(y * y + 1)) * sin(0.5 * (M_PI - atan(y)));

    }

	// Randpunkte setzen, die von 0 bis -1 laufen
	#pragma omp parallel for
	for (int i = n[l - 1] / 2 - 1; i > 0 - 1; i--) {
		double y = -1.0 + i * h[l - 1];

		// rechte Seite unten
		u[l - 1](n[l - 1] - 1, i) = sqrt(sqrt(y * y + 1))
				* sin(0.5 * (2 * M_PI - atan(-y)));

		// obere Seite links
		u[l - 1](i, n[l - 1] - 1) = sqrt(sqrt(y * y + 1))
				* sin(0.5 * (0.5 * M_PI + atan(-y)));

		// untere Seite links
		u[l - 1](i, 0) = sqrt(sqrt(y * y + 1))
				* sin(0.5 * (((3.0 / 2.0) * M_PI) - atan(-y)));

		// linke Seite unten
		u[l - 1](0, i) = sqrt(sqrt(y * y + 1)) * sin(0.5 * (M_PI + atan(-y)));
	}
    
    for (int j = 0; j < n[l - 1]; j++) {
        for (int i = 0; i < n[l - 1]; i++) {
            u_neu[l-1](i,j) = u[l-1](i,j);
        }
    }

    //init.dat Ausgabe
    std::ofstream init;
    init.open("init.dat", std::ios::out);
    
    // Ausgabe fuer solution.dat
    init << "# x y u(x,y)\n" << std::endl;
    for (int j = 0; j < n[l - 1]; j++) {
        for (int i = 0; i < n[l - 1]; i++) {
            double x = -1.0 + i * h[l - 1];
            double y = -1.0 + j * h[l - 1];
            init << x << " " << y << " " << u[l - 1](i, j) << std::endl;
        }
        init << std::endl;
    }
    init.close();


	// Multigrid solver ------------------------------------------------------------------------------------------------
	std::cout << "Your Alias: " << "bu43jazu" << std::endl;
	struct timeval t0, t;
	gettimeofday(&t0, NULL);
    //Red_Black_Gauss( n[l-1], u[l-1], f[l-1], h[l-1], 10000);
    for(int i = 0; i < cycle; i++){
        solveMG(l - 1, u , u_neu, f, n, h, res, v1, v2, gamma);
    }
	gettimeofday(&t, NULL);
	std::cout << "Wall clock time of MG execution: "
			<< ((int64_t) (t.tv_sec - t0.tv_sec) * (int64_t) 1000000
					+ (int64_t) t.tv_usec - (int64_t) t0.tv_usec) * 1e-3
			<< " ms" << std::endl;

	// output --------------------------------------------------------------------------------------------------------------





    std::cout << "L2 residual: "  << residuum(n[l-1],  u[l-1], f[l-1], h[l-1]) << std::endl;


	std::ofstream out;
	out.open("solution.dat", std::ios::out);

	// Ausgabe fuer solution.dat
	out << "# x y u(x,y)\n" << std::endl;
	for (int j = 0; j < n[l - 1]; j++) {
		for (int i = 0; i < n[l - 1]; i++) {
			double x = -1.0 + i * h[l - 1];
			double y = -1.0 + j * h[l - 1];
			out << x << " " << y << " " << u[l - 1](i, j) << std::endl;
		}
		out << std::endl;
	}
    out.close();

    double error = 0.0;
    residual(n[l-1], u[l-1], f[l-1], res[l-1], h[l-1]);
    for(int i = 0; i< n[l-1]; i++){
        for(int j = 0; j < n[l-1]; j++){
            u[l-1](i,j)=0.0;
        }
    }

    Red_Black_Gauss(n[l-1], u[l-1], res[l-1], h[l-1], 1000);
    for(int i = 0; i< n[l-1]; i++){
        for(int j = 0; j < n[l-1]; j++){
            error += u[l-1](i,j)*u[l-1](i,j);
        }
    }

    std::cout << std::fixed << "L2 error: " << std::fixed << sqrt( error/(n[l-1]-1)*(n[l-1]-1)) << std::endl;

}
