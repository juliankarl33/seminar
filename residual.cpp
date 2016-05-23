/*
// calculates the residual "res=Au-f"
void residual(int n, grid<type> &u, grid<type> &f, grid<type> &res, double h,
		int l) {

	double h2 = h * h;
	__m128d links, rechts, mitte, oben, unten, rightHandSide, erg1, erg2;
	__m128d const4 = _mm_set_pd(4, 4);
	__m128d meshsize = _mm_load_pd1(&h2);

	//#pragma omp parallel for private(links, rechts, mitte, oben, unten, rightHandSide, erg1, erg2)
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

			//_mm_storeu_pd(&res(k, i), erg1);
			//fprintf(stderr, "noch nicht\n");
			_mm_stream_pd(&res(k, i), erg1); //eventuell non temporal store benoetigt aber allignment
			//fprintf( stderr, "aus\n");

		}
	}
}
*/

// calculates the residual "res=Au-f"
void residual(int n, grid<type> &u, grid<type> &f, grid<type> &res, double h,
		int l) {

	//#pragma omp parallel for
	for (int i = 1; i < n - 1; i++) {

		for (int k = 1; k < n - 1; k = k + 2) {
			if (!(i == n / 2 && k >= n / 2)) {
			res(k, i) = ((u(k-1, i) + u(k+1, i)) + (u(k, i - 1) + u(k, i + 1))
					- (4 * u(k, i))) / (h * h) + f(k, i);
		}}
	}
}
