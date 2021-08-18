#define extract_cxx
#include "extract.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
using namespace std;

void extract::Loop()
{
//My macro to extract the data
        TFile *f=new TFile("/path/to/heysham_2_file.root");
        TTree *tr=(TTree*)f->Get("data;1");

        Double_t a, b, d, g, h, j, k, l,m, n, o, p, q, r, s, t, u, v, w, z;
        Int_t c, e;

        tr->SetBranchAddress("n100",&a);
        tr->SetBranchAddress("n100_prev",&b);
        tr->SetBranchAddress("dt_prev_us",&d);
        tr->SetBranchAddress("inner_hit",&c);
        tr->SetBranchAddress("inner_hit_prev",&e);
        tr->SetBranchAddress("beta_one", &g);
        tr->SetBranchAddress("beta_one_prev", &h);
        tr->SetBranchAddress("beta_two", &j);
        tr->SetBranchAddress("beta_two_prev", &k);
        tr->SetBranchAddress("beta_three", &l);
        tr->SetBranchAddress("beta_three_prev", &m);
        tr->SetBranchAddress("beta_four", &n);
        tr->SetBranchAddress("beta_four_prev", &o);
	tr->SetBranchAddress("beta_five", &p);
        tr->SetBranchAddress("beta_five_prev", &q);
        tr->SetBranchAddress("beta_six", &r);
        tr->SetBranchAddress("beta_six_prev", &s);
        tr->SetBranchAddress("good_pos", &t);
        tr->SetBranchAddress("good_pos_prev", &u);
        tr->SetBranchAddress("closestPMT", &v);
	tr->SetBranchAddress("closestPMT_prev", &w);
	tr->SetBranchAddress("drPrevr", &z);

        ofstream file1;
        file1.open("/path/to/heysham_2_file.txt");
        for (Int_t i=0; i<tr->GetEntries(); i++){
                tr->GetEntry(i);
                file1 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
	}
	file1.close();

        TFile *f2=new TFile("/path/to/torness_file.root");
        TTree *tr2=(TTree*)f2->Get("data;1");

        tr2->SetBranchAddress("n100",&a);
        tr2->SetBranchAddress("n100_prev",&b);
        tr2->SetBranchAddress("dt_prev_us",&d);
        tr2->SetBranchAddress("inner_hit",&c);
        tr2->SetBranchAddress("inner_hit_prev",&e);
        tr2->SetBranchAddress("beta_one", &g);
        tr2->SetBranchAddress("beta_one_prev", &h);
        tr2->SetBranchAddress("beta_two", &j);
        tr2->SetBranchAddress("beta_two_prev", &k);
        tr2->SetBranchAddress("beta_three", &l);
        tr2->SetBranchAddress("beta_three_prev", &m);
        tr2->SetBranchAddress("beta_four", &n);
        tr2->SetBranchAddress("beta_four_prev", &o);
        tr2->SetBranchAddress("beta_five", &p);
        tr2->SetBranchAddress("beta_five_prev", &q);
        tr2->SetBranchAddress("beta_six", &r);
        tr2->SetBranchAddress("beta_six_prev", &s);
        tr2->SetBranchAddress("good_pos", &t);
        tr2->SetBranchAddress("good_pos_prev", &u);
        tr2->SetBranchAddress("closestPMT", &v);
        tr2->SetBranchAddress("closestPMT_prev", &w);
        tr2->SetBranchAddress("drPrevr", &z);
        ofstream file2;
        file2.open("/path/to/torness_file.txt");
        for (Int_t i=0; i<tr2->GetEntries(); i++){
                tr2->GetEntry(i);
		file2 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
	}
        file2.close();
        TFile *f3=new TFile("/path/to/world_file.root");
        TTree *tr3=(TTree*)f3->Get("data;1");
        tr3->SetBranchAddress("n100",&a);
        tr3->SetBranchAddress("n100_prev",&b);
        tr3->SetBranchAddress("dt_prev_us",&d);
        tr3->SetBranchAddress("inner_hit",&c);
        tr3->SetBranchAddress("inner_hit_prev",&e);
        tr3->SetBranchAddress("beta_one", &g);
        tr3->SetBranchAddress("beta_one_prev", &h);
        tr3->SetBranchAddress("beta_two", &j);
        tr3->SetBranchAddress("beta_two_prev", &k);
        tr3->SetBranchAddress("beta_three", &l);
        tr3->SetBranchAddress("beta_three_prev", &m);
        tr3->SetBranchAddress("beta_four", &n);
        tr3->SetBranchAddress("beta_four_prev", &o);
        tr3->SetBranchAddress("beta_five", &p);
        tr3->SetBranchAddress("beta_five_prev", &q);
        tr3->SetBranchAddress("beta_six", &r);
        tr3->SetBranchAddress("beta_six_prev", &s);
        tr3->SetBranchAddress("good_pos", &t);
        tr3->SetBranchAddress("good_pos_prev", &u);
        tr3->SetBranchAddress("closestPMT", &v);
        tr3->SetBranchAddress("closestPMT_prev", &w);
        tr3->SetBranchAddress("drPrevr", &z);
        ofstream file3;
        file3.open("/path/to/world_file.txt");
        for (Int_t i=0; i<tr3->GetEntries(); i++){
                tr3->GetEntry(i);
                file3 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
        }
        file3.close();

        TFile *f4=new TFile("/path/to/neutrons_file.root");
        TTree *tr4=(TTree*)f4->Get("data;1");

        tr4->SetBranchAddress("n100",&a);
        tr4->SetBranchAddress("n100_prev",&b);
        tr4->SetBranchAddress("dt_prev_us",&d);
        tr4->SetBranchAddress("inner_hit",&c);
        tr4->SetBranchAddress("inner_hit_prev",&e);
        tr4->SetBranchAddress("beta_one", &g);
        tr4->SetBranchAddress("beta_one_prev", &h);
        tr4->SetBranchAddress("beta_two", &j);
        tr4->SetBranchAddress("beta_two_prev", &k);
        tr4->SetBranchAddress("beta_three", &l);
        tr4->SetBranchAddress("beta_three_prev", &m);
        tr4->SetBranchAddress("beta_four", &n);
        tr4->SetBranchAddress("beta_four_prev", &o);
        tr4->SetBranchAddress("beta_five", &p);
        tr4->SetBranchAddress("beta_five_prev", &q);
        tr4->SetBranchAddress("beta_six", &r);
        tr4->SetBranchAddress("beta_six_prev", &s);
        tr4->SetBranchAddress("good_pos", &t);
        tr4->SetBranchAddress("good_pos_prev", &u);
        tr4->SetBranchAddress("closestPMT", &v);
        tr4->SetBranchAddress("closestPMT_prev", &w);
        tr4->SetBranchAddress("drPrevr", &z);
       	ofstream file4;
        file4.open("/path/to/neutrons_file.txt");
        for (Int_t i=0; i<tr4->GetEntries(); i++){
                tr4->GetEntry(i);
                file4 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
        }
        file4.close();
        TFile *f5=new TFile("/path/to/geoneutrinos_file.root");
        TTree *tr5=(TTree*)f5->Get("data;1");
        tr5->SetBranchAddress("n100",&a);
        tr5->SetBranchAddress("n100_prev",&b);
        tr5->SetBranchAddress("dt_prev_us",&d);
        tr5->SetBranchAddress("inner_hit",&c);
        tr5->SetBranchAddress("inner_hit_prev",&e);
        tr5->SetBranchAddress("beta_one", &g);
        tr5->SetBranchAddress("beta_one_prev", &h);
        tr5->SetBranchAddress("beta_two", &j);
        tr5->SetBranchAddress("beta_two_prev", &k);
        tr5->SetBranchAddress("beta_three", &l);
        tr5->SetBranchAddress("beta_three_prev", &m);
        tr5->SetBranchAddress("beta_four", &n);
        tr5->SetBranchAddress("beta_four_prev", &o);
        tr5->SetBranchAddress("beta_five", &p);
        tr5->SetBranchAddress("beta_five_prev", &q);
        tr5->SetBranchAddress("beta_six", &r);
        tr5->SetBranchAddress("beta_six_prev", &s);
        tr5->SetBranchAddress("good_pos", &t);
        tr5->SetBranchAddress("good_pos_prev", &u);
        tr5->SetBranchAddress("closestPMT", &v);
        tr5->SetBranchAddress("closestPMT_prev", &w);
        tr5->SetBranchAddress("drPrevr", &z);
       	ofstream file5;
        file5.open("/path/to/geoneutrinos_file.txt");
        for (Int_t i=0; i<tr5->GetEntries(); i++){
                tr5->GetEntry(i);
                file5 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
        }
        file5.close();
        TFile *f6=new TFile("/path/to/li9_file.root");
        TTree *tr6=(TTree*)f6->Get("data;1");
        tr6->SetBranchAddress("n100",&a);
        tr6->SetBranchAddress("n100_prev",&b);
        tr6->SetBranchAddress("dt_prev_us",&d);
        tr6->SetBranchAddress("inner_hit",&c);
        tr6->SetBranchAddress("inner_hit_prev",&e);
        tr6->SetBranchAddress("beta_one", &g);
        tr6->SetBranchAddress("beta_one_prev", &h);
        tr6->SetBranchAddress("beta_two", &j);
        tr6->SetBranchAddress("beta_two_prev", &k);
        tr6->SetBranchAddress("beta_three", &l);
        tr6->SetBranchAddress("beta_three_prev", &m);
        tr6->SetBranchAddress("beta_four", &n);
        tr6->SetBranchAddress("beta_four_prev", &o);
        tr6->SetBranchAddress("beta_five", &p);
        tr6->SetBranchAddress("beta_five_prev", &q);
        tr6->SetBranchAddress("beta_six", &r);
        tr6->SetBranchAddress("beta_six_prev", &s);
        tr6->SetBranchAddress("good_pos", &t);
        tr6->SetBranchAddress("good_pos_prev", &u);
        tr6->SetBranchAddress("closestPMT", &v);
        tr6->SetBranchAddress("closestPMT_prev", &w);
        tr6->SetBranchAddress("drPrevr", &z);
       	ofstream file6;
        file6.open("/path/to/li9_file.txt");
        for (Int_t i=0; i<tr6->GetEntries(); i++){
                tr6->GetEntry(i);
                file6 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
        }
        file6.close();
        TFile *f7=new TFile("/path/to/n17_file.root");
        TTree *tr7=(TTree*)f7->Get("data;1");
        tr7->SetBranchAddress("n100",&a);
        tr7->SetBranchAddress("n100_prev",&b);
        tr7->SetBranchAddress("dt_prev_us",&d);
        tr7->SetBranchAddress("inner_hit",&c);
        tr7->SetBranchAddress("inner_hit_prev",&e);
        tr7->SetBranchAddress("beta_one", &g);
        tr7->SetBranchAddress("beta_one_prev", &h);
        tr7->SetBranchAddress("beta_two", &j);
        tr7->SetBranchAddress("beta_two_prev", &k);
        tr7->SetBranchAddress("beta_three", &l);
        tr7->SetBranchAddress("beta_three_prev", &m);
        tr7->SetBranchAddress("beta_four", &n);
        tr7->SetBranchAddress("beta_four_prev", &o);
        tr7->SetBranchAddress("beta_five", &p);
        tr7->SetBranchAddress("beta_five_prev", &q);
        tr7->SetBranchAddress("beta_six", &r);
        tr7->SetBranchAddress("beta_six_prev", &s);
        tr7->SetBranchAddress("good_pos", &t);
        tr7->SetBranchAddress("good_pos_prev", &u);
        tr7->SetBranchAddress("closestPMT", &v);
        tr7->SetBranchAddress("closestPMT_prev", &w);
        tr7->SetBranchAddress("drPrevr", &z);
       	ofstream file7;
        file7.open("/path/to/n17_file.txt");
        for (Int_t i=0; i<tr7->GetEntries(); i++){
                tr7->GetEntry(i);
                file7 << a << " " << b << " " << d << " " << c << " " << e << " " << g << " " << h << " " << j << " " << k << " " << l << " " << m << " " << n << " " << o << " " << p << " " << q << " " << r << " " << s << " " << t << " " << u << " " << v << " " << w << " " << z <<"\n";
        }
        file7.close();
}
