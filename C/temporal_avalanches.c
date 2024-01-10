#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// define sets of parameters:
// rough exploration showed that the critical value of Bc is around 4.0434647
// for a system with N=10000 nodes with K=7 neighbours each (this is just for reference).
// we simulate for a given time_depth and throw away the first throw_away time steps.

#define N 10000
#define K 7
#define time_depth 10100
#define throw_away 100
#define B 4.0502856
// rough Bc 4.0434647


// linked list structure to keep track of the persistence statistics
// value is the persistence time (length of the avalanche)
// mag is the magnitude of the avalanche (size of the avalanche)
// then there is a pointer to the next avalanche statistics object
// this is done because we don't know a priori the number of avalanches (hence why we don't use vectors)
typedef struct persistence {
  int val;
  double mag;
  struct persistence *next;
} persistence;


// initialise the linked list pointers
int count=0;
persistence *init_ptr,*last_ptr,*tmp_ptr;

void generate_time_series();
void compute_persistence_stat();
void calc_corr();

/**
 * @file code.c
 * @brief This file contains the main function.
 * We first generate a time series of the total delay of the system.
 * Then we compute the persistence statistics and the autocorrelation function.
 * Finally, we compute the autocorrelation function of the time-series.
 * Results are stored in files "persistence" (persistence statistics), "avalanche" (avalanche size statistics),
 * "correlations" (autocorrelation function of the total delay of the system).
 * 
 */
int main() {

  generate_time_series();
  compute_persistence_stat();
  calc_corr();

  return 0;
}

/**
 * @brief Generates a time series.
 * 
 * This function generates a time series consisting of the 
 * mean delay of the system at each time step and stores it into files.
 * At each time step, we update a node's delay as delay[i] = max{delay[j]-B} + eps[i],
 * where the max is taken over the K neighbours of node i, and eps[i] is a random variable with
 * exponential distribution with mean 1 (computed as the log of a uniform random variable).
 * At each time step, the mean delay of the system is computed and stored into the file.
 * Total simulation length is time_depth, and we throw away the first throw_away time steps.
 * @return void
 */
void generate_time_series() {

  int i,j,k,l,firms[K],chosen,tf;
  double x,delay[N],shadow[N],maxdel,totdel;
  char name[100];
  FILE *fpw1;

  sprintf(name,"out");
  fpw1 = fopen(name,"w");

  totdel = 0.0;
  // initialise array
  for (i = 0; i < N; i++) {
    x = drand48();
    delay[i] = -log(x);
    totdel += delay[i]-B;
  }
  // fprintf(fp,"%d\t%12.9lf\n",0,totdel/((double)(N)));

  // for every time depth repeat
  for (i = 1; i <= time_depth; i++) {
    
    // first copy the existing delays to a shadow array
    for (j = 0; j < N; j++)
      shadow[j] = delay[j];

    totdel = 0.0;
    
    for (j = 0; j < N; j++) {
      
      // choose K random different firms at random
      chosen = 0;
      while (chosen < K) {
	if (chosen == 0) {
	  x = drand48();
	  firms[chosen++] = (int)(((double)(N))*x);
	}
	else {
	  tf = 0;
	  while (tf == 0) {
	    x = drand48();
	    k = (int)(((double)(N))*x);
	    tf = 1;
	    // check if j is already chosen
	    for (l = 0; l < chosen; l++)
	      if (k != firms[l] && tf == 1)
		tf = 1;
	      else
		tf = 0;
	  }
	  firms[chosen++] = k;
	}
      }

      // max delay of the chosen firms min buffer
      // as max(delays - B, 0)
      maxdel = shadow[firms[0]] - B;
      for (k = 1; k < K; k++)
	if (maxdel < shadow[firms[k]] - B)
	  maxdel = shadow[firms[k]] - B;
      if (maxdel < 0.0)
	maxdel = 0.0;

      // new delay for firm j
      delay[j] = maxdel;
      x = drand48();
      // adding exponentially distributed (Exp(1)) random noise
      delay[j] += -log(x);

      // update the total delay, defined as sum_i (delay[i]-B)
      totdel += delay[j]-B;

    }
    // save values after the burn-in period
    if (i >= throw_away) {
      fprintf(fpw1,"%12.9lf\n",totdel/((double)(N)));
    }
    
  }

  fclose(fpw1);
  return;

}

/**
 * Computes the persistence statistics.
 */
void compute_persistence_stat() {

  int i,p_len,time_series_len;
  double p,p_val;
  char name[100];
  FILE *fpr,*fpw,*fpw1;

  time_series_len = time_depth-throw_away+1;

  // open the time-series file
  sprintf(name,"out");
  fpr = fopen(name,"r");

  // open files to store the persistence and avalanche size statistics
  sprintf(name,"persistence");
  fpw = fopen(name,"w");
  sprintf(name,"avalanche");
  fpw1 = fopen(name,"w");

  i = 0;
  fscanf(fpr,"%le",&p);

  // find the first negative value in the time-series
  while (p > 0 && i < time_series_len) {
    ++i;
    fscanf(fpr,"%le",&p);
  }
    
  // then find the first positive value in the series
  while (p < 0 && i < time_series_len) {
    ++i;
    fscanf(fpr,"%le",&p);
  }

  // the previous makes sure that i represents the time-step for which the first avalanche starts
  // (because it's possible we had avalanches before the burn-in period in the time series generation)


  // we parse the entire time-series
  while (i < time_series_len) {

    // check how long the positive value holds on
    p_len = 1;
    p_val = p;


    // find the first negative value in the series
    // the time-series will be positive for p_val consecutive steps
    // the size of the avalanche is the sum of all values p, which are the values of the total delay which have total_delay > B
    while (p > 0 && i < time_series_len) {
      ++i;
      fscanf(fpr,"%le",&p);
      if (p > 0) {
	++p_len;
        p_val += p;
      }
    }

    // at this point p_val has the size of the avalanche
    // p_len has the persistence time


    // update the avalanche count and the persistence length of the avalnche
    // this is done by filling in the linked list for persistence (See the struct defined at the beginning of the file)
    if (i < time_series_len) {
      ++count;
      if (count == 1) {
	init_ptr = last_ptr = (persistence *) malloc(sizeof(persistence));
	last_ptr->val = p_len;
        last_ptr->mag = p_val;
	last_ptr->next = NULL;
      }
      else {
	tmp_ptr = last_ptr;
	last_ptr = (persistence *) malloc(sizeof(persistence));
	last_ptr->val = p_len;
        last_ptr->mag = p_val;
	last_ptr->next = NULL;
	tmp_ptr->next = last_ptr;
      }
    }
      
    // find the time-step corresponding to the beginning of the next avalanche
    while (p < 0 && i < time_series_len) {
      ++i;
      fscanf(fpr,"%le",&p);
    }


    // save the persistence and avalanche size statistics
    fprintf(fpw,"%d\t%d\n",count,p_len);
    fprintf(fpw1,"%12.9lf\n",p_val);
    
  }

  fclose(fpr);
  fclose(fpw);
  fclose(fpw1);

  return;
}



/**
 * Calculates lagged autocorrelation function between time-series x and y.
  * in this case, time-series x is the time-series of the total delay of the system
  * while y is the time-series of the total delay of the system with a lag of lag time-steps
 */
void calc_corr() {

  int i,lag,time_series_len;
  double x,y,corr,sumx,sumy,sumxy,sumxsq,sumysq;
  double xmean,ymean,xymean,xsqmean,ysqmean;
  char name[100];
  FILE *fpr1,*fpr2,*fpw;

  time_series_len = time_depth-throw_away+1;
  // open the file where we will store the autocorrelation function
  sprintf(name,"correlations");
  fpw = fopen(name,"w");
  fprintf(fpw,"%d\t%12.9lf\n",0,1.0);  
 

  // iterate over lag values, jumping by 20 time-steps
  // max lag is 50000 time-steps (arbitrary)
  for (lag = 1; lag < 50000; lag+=20) {
    // read the time-series files
    sprintf(name,"out");
    fpr1 = fopen(name,"r");
    fpr2 = fopen(name,"r");


    // compute standard Pearson correlation function
    // computes \sum_i y_i * y_{i+lag} / \sum_i y_i^2
    // where y_i = x_i - \bar{x} 
    fscanf(fpr1,"%le",&x);
    for (i = 0; i <= lag; i++)
      fscanf(fpr2,"%le",&y);
    sumx = x; sumy = y; sumxy = x*y; sumxsq = x*x; sumysq = y*y; count = 1;

    for (i = 0; i < time_series_len - lag - 1; i++) {
      fscanf(fpr1,"%le",&x);
      fscanf(fpr2,"%le",&y);
      sumx += x; sumy += y; sumxy += x*y; sumxsq += x*x; sumysq += y*y; ++count;
    }
    
    fclose(fpr1);
    fclose(fpr2);

    xmean = sumx/((double)(count));
    ymean = sumy/((double)(count));
    xymean = sumxy/((double)(count));
    xsqmean = sumxsq/((double)(count));
    ysqmean = sumysq/((double)(count));

    // compute the correlation
    corr = (xymean-xmean*ymean)/(sqrt(xsqmean-xmean*xmean)*sqrt(ysqmean-ymean*ymean));

    // save the correlation to the file on a line with tab separated values
    // first value is the lag, second value is the correlation
    fprintf(fpw,"%d\t%12.9lf\n",lag,corr);
    
  }
    
  fclose(fpw);

  return;
}
