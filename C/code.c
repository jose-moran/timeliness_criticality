#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// define sets of parameters:
// rough exploration showed that the critical value of Bc is around 4.0434647
// for a system with N nodes with K neighbours each.

#define N 10000
#define K 7
#define time_depth 10100000
#define throw_away 100000
#define B 4.0502856
// rough Bc 4.0434647

typedef struct persistence {
  int val;
  double mag;
  struct persistence *next;
} persistence;

int count=0;
persistence *init_ptr,*last_ptr,*tmp_ptr;

void generate_time_series();
void compute_persistence_stat();
void persistence_stat(int a);
void calc_corr();

int main() {

  long now;

  now = time(NULL)+(long)(num);
  srand48(now);

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
  FILE *fpw1,*fpw2;

  sprintf(name,"%d/out",num);
  fpw1 = fopen(name,"w");
  sprintf(name,"%d/out_pos",num);  
  fpw2 = fopen(name,"w");

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
      
      // choose K random different firms
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
      maxdel = shadow[firms[0]] - B;
      for (k = 1; k < K; k++)
	if (maxdel < shadow[firms[k]] - B)
	  maxdel = shadow[firms[k]] - B;
      if (maxdel < 0.0)
	maxdel = 0.0;

      // new delay for firm i
      delay[j] = maxdel;
      x = drand48();
      delay[j] += -log(x);

      totdel += delay[j]-B;

    }

    if (i >= throw_away) {
      fprintf(fpw1,"%12.9lf\n",totdel/((double)(N)));
      if (totdel > 0)
	fprintf(fpw2,"%12.9lf\n",totdel/((double)(N)));
    }
    
  }

  fclose(fpw1);
  fclose(fpw2);
  return;

}

void compute_persistence_stat() {

  int i,p_len,time_series_len;
  double p,p_val;
  char name[100];
  FILE *fpr,*fpw,*fpw1;

  time_series_len = time_depth-throw_away+1;
  sprintf(name,"%d/out",num);
  fpr = fopen(name,"r");
  sprintf(name,"%d/persistence",num);
  fpw = fopen(name,"w");
  sprintf(name,"%d/avalanche",num);
  fpw1 = fopen(name,"w");

  i = 0;
  fscanf(fpr,"%le",&p);

  // find the first negative mean delay
  while (p > 0 && i < time_series_len) {
    ++i;
    fscanf(fpr,"%le",&p);
  }
    
  // then find the first positive value in the series
  while (p < 0 && i < time_series_len) {
    ++i;
    fscanf(fpr,"%le",&p);
  }

  while (i < time_series_len) {

    // check how long the positive value holds on
    p_len = 1;
    p_val = p;

    // find the first negative value in the series
    while (p > 0 && i < time_series_len) {
      ++i;
      fscanf(fpr,"%le",&p);
      if (p > 0) {
	++p_len;
        p_val += p;
      }
    }

    // unpdate the avalanche count and the persistence length of the avalnche
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
      
    // find the next positive value in the series
    while (p < 0 && i < time_series_len) {
      ++i;
      fscanf(fpr,"%le",&p);
    }

    fprintf(fpw,"%d\t%d\n",count,p_len);
    fprintf(fpw1,"%12.9lf\n",p_val);
    
  }

  fclose(fpr);
  fclose(fpw);
  fclose(fpw1);

  //persistence_stat(count);

  return;
}

void calc_corr() {

  int i,lag,time_series_len;
  double x,y,corr,sumx,sumy,sumxy,sumxsq,sumysq;
  double xmean,ymean,xymean,xsqmean,ysqmean;
  char name[100];
  FILE *fpr1,*fpr2,*fpw;

  time_series_len = time_depth-throw_away+1;
  sprintf(name,"%d/correlations",num);
  fpw = fopen(name,"w");
  fprintf(fpw,"%d\t%12.9lf\n",0,1.0);  
 
  for (lag = 1; lag < 42185 /* time_series_len */; lag+=20) {
    
    sprintf(name,"%d/out",num);
    fpr1 = fopen(name,"r");
    fpr2 = fopen(name,"r");

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

    corr = (xymean-xmean*ymean)/(sqrt(xsqmean-xmean*xmean)*sqrt(ysqmean-ymean*ymean));

    fprintf(fpw,"%d\t%12.9lf\n",lag,corr);
    
  }
    
  fclose(fpw);

  return;
}

void persistence_stat(int n) {

  int i;
  double a,mean=0.0,meansq=0.0;
  char name[100];
  FILE *fpw;
  
  sprintf(name,"%d/persistence",num);
  fpw = fopen(name,"a");

  tmp_ptr = init_ptr;
  for (i = 0; i < n; i++) {
    a = (double)(tmp_ptr->val);
    mean += a;
    meansq += a*a;
    tmp_ptr = tmp_ptr->next;
  }
  mean /= ((double)(n));
  meansq /= ((double)(n));

  fprintf(fpw,"%12.9lf\t%12.9lf\n",mean,sqrt(meansq-mean*mean));
  fclose(fpw);
  
  return;
}
