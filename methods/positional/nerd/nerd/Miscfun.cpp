#include "NERD.h"

const gsl_rng_type * gsl_T;
 gsl_rng * gsl_r;
 
 void rand_setup()
 {
     gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);
 }
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}


double randomLoc(double size)
{
  double loc = (double) size * gsl_rng_uniform(gsl_r);
     //int loc = (int)gsl_rng_uniform(gsl_r,size);
     return loc;
}
int randomIntLoc(int size)
{
int loc=gsl_rng_uniform_int (gsl_r,size);
return loc;

}
double coinThrow ()
{
    return  gsl_rng_uniform(gsl_r);
}
