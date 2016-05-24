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
		//data = NULL;
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
		//data = (double*)malloc((xDim+1)*yDim);
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
    file << "Wall clock time of " << text << "\t \t \t"
            << ((int64_t) (t1.tv_sec - t0.tv_sec) * (int64_t) 1000000
                    + (int64_t) t1.tv_usec - (int64_t) t0.tv_usec) * 1e-3
            << " ms" << "\t" <<", level: " << level << std::endl;
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

  
  __m128d const2 = _mm_set_pd(2, 2);
  __m128d const4 = _mm_set_pd(4, 4);
  __m128d const1over16 = _mm_set_pd(0.0625, 0.0625);
 // __m128d reg1, reg2, reg3, reg4, reg5;
   __m128d reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, erg;
   
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
			  
			  reg1 = _mm_loadu_pd(&from(2 * j - 1,  2 * i + 1));
			  reg2 = _mm_loadu_pd(&from(2 * j, 	2 * i + 1));
			  reg3 = _mm_loadu_pd(&from(2 * j + 1,  2 * i + 1));
			  reg4 = _mm_loadu_pd(&from(2 * j - 1,  2 * i));
			  reg5 = _mm_loadu_pd(&from(2 * j,      2 * i));
			  reg6 = _mm_loadu_pd(&from(2 * j + 1,  2 * i));
			  reg7 = _mm_loadu_pd(&from(2 * j - 1,  2 * i - 1));
			  reg8 = _mm_loadu_pd(&from(2 * j,      2 * i - 1));
			  reg9 = _mm_loadu_pd(&from(2 * j + 1,  2 * i - 1));
			  
			  reg2 = _mm_mul_pd(reg2, const2);			
			  reg4 = _mm_mul_pd(reg4, const2);			
			  reg6 = _mm_mul_pd(reg6, const2);			
			  reg8 = _mm_mul_pd(reg8, const2);	
			  
			  reg5 = _mm_mul_pd(reg5, const4);
			  
			  erg = _mm_add_pd(reg1, reg2);
			  erg = _mm_add_pd(erg, reg3);
			  erg = _mm_add_pd(erg, reg4);
			  erg = _mm_add_pd(erg, reg5);
			  erg = _mm_add_pd(erg, reg6);
			  erg = _mm_add_pd(erg, reg7);
			  erg = _mm_add_pd(erg, reg8);
			  erg = _mm_add_pd(erg, reg9);
				
			  erg = _mm_mul_pd(erg, const1over16);
				
			  _mm_stream_pd(&to(j, i), erg); 
			  
			  	reg1 = _mm_loadu_pd(&from(2 * j - 1, 2 * i + 1));	// links-oben
				reg2 = _mm_loadu_pd(&from(2 * j, 2 * i + 1));		// oben
			  
				reg3 = _mm_mul_pd(reg2, const2);			// oben x 2
				reg4 = _mm_add_pd(reg3, reg1);				// links-oben + 2x oben
			  
				reg5 = _mm_loadu_pd(&from(2 * j + 1, 2 * i + 1));	// rechts-oben
				reg2 = _mm_add_pd(reg4, reg5);				// links-oben + 2x oben
				
				reg1 = _mm_loadu_pd(&from(2 * j - 1, 2 * i));		// links
				reg3 = _mm_mul_pd(reg1, const2);			// links x2
				reg1 = _mm_loadu_pd(&from(2 * j, 2 * i));		// mitte
				reg4 = _mm_mul_pd(reg1, const4);			// mitte x4
				reg1 = _mm_add_pd(reg4, reg3);				// mitte x4 + links x2
				reg3 = _mm_add_pd(reg1, reg2);				// links-oben + 2x oben + mitte x4 + links x2
				
				reg1 = _mm_loadu_pd(&from(2 * j + 1, 2 * i));		// rechts
				reg2 = _mm_loadu_pd(&from(2 * j - 1, 2 * i - 1));		// links-unten
			  
				reg4 = _mm_mul_pd(reg1, const2);			// rechts x 2
				reg1 = _mm_add_pd(reg4, reg2);				// links-unten + 2x rechts
				
				
				reg2 = _mm_loadu_pd(&from(2 * j, 2 * i - 1));		// unten
				reg5 = _mm_loadu_pd(&from(2 * j + 1, 2 * i - 1));		// rechts-unten
			  
				reg4 = _mm_mul_pd(reg2, const2);			// unten x 2
				reg2 = _mm_add_pd(reg5, reg4);				// rechts-unten + 2x unten
				
				reg5 = _mm_add_pd(reg1, reg2);				// rechts-unten + 2x unten + links-unten + 2x rechts
				
				reg1 = _mm_add_pd(reg3, reg5);	
				
				reg2 = _mm_mul_pd(reg1, const1over16);
				
				_mm_stream_pd(&to(j, i), reg2); 
				
			}
		}
	}
}
*/

void interpolation(int l, grid<type>& from, grid<type>& to, intVec& n) {

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

//Parameter "finest" wird fuer die Neuman-RB (GHOST) benoetigt. Damit wird nur beim feinsten Grid die Neumann-RB im Red-Black-Gauss berechnet
void solveMG(int l, std::vector<grid<type>>& u, std::vector<grid<type>>& f,
		intVec& n, std::vector<type>& h, std::vector<grid<type>>& res, int v1 =
                3, int v2 = 2, int gamma = 2) {

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
        print_time("RBGauss presmoothing ", t0, t1, l, messung);
        print_time("Residuum ", t1, t2, l, messung);
        print_time("Restrict residual ", t2, t3, l, messung);
        print_time("set u = 0 ", t3, t5, l, messung);
        print_time("interpolation ", t6, t7, l, messung);
        print_time("correction ", t7, t8, l, messung);
        print_time("RBGauss postsmoothing ", t9, t10, l, messung);
    }
}



int main(int argc, char **argv) {

	// Ueberpruefung, ob Eingabeparamter passen
	if (argc != 2) {
		fprintf(stderr, "Usage: ./mgsolve levels\n");
		exit(EXIT_SUCCESS);
	}

	// definitions
	int l = atoi(argv[1]);	// number of levels

	intVec n(l, 0);	// total number of gird points in x and y-direction
	typeVec h(l, 0.0);// mesh size of each levels, where h[0] is the mesh size of the coarsesed grid
    std::vector<grid<type>> f(l);	// vector of grids for the rig1ht hand side
	std::vector<grid<type>> res(l);	// vector of grids for the residuums
	std::vector<grid<type>> u(l);	// vector of grids for the approximation u

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

	// Horizontale und vertikale Randpunkte setzen
	u[l - 1](n[l - 1] - 1, n[l - 1] / 2) = 0.0;	// Rechts
	u[l - 1](0, n[l - 1] / 2) = 1.0;	// Links
	u[l - 1](n[l - 1] / 2, n[l - 1] - 1) = sin(0.25 * M_PI);	// Oben
	u[l - 1](n[l - 1] / 2, 0) = sin(0.75 * M_PI);	// Unten

	// Randpunkte setzen, die von 0 bis 1 laufen
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
    for(int i = 0; i < 15; i++){
        solveMG(l - 1, u, f, n, h, res);
    }
	gettimeofday(&t, NULL);
	std::cout << "Wall clock time of MG execution: "
			<< ((int64_t) (t.tv_sec - t0.tv_sec) * (int64_t) 1000000
					+ (int64_t) t.tv_usec - (int64_t) t0.tv_usec) * 1e-3
			<< " ms" << std::endl;

	// output --------------------------------------------------------------------------------------------------------------





    std::cout << "L2 residual: " << residuum(n[l-1],  u[l-1], f[l-1], h[l-1]) << std::endl;


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

     std::cout << "L2 error: " << sqrt( error/(n[l-1]-1)*(n[l-1]-1)) << std::endl;

}
