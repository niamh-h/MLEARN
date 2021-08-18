//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Jul 19 14:46:24 2021 by ROOT version 6.18/05
// from TTree data/low-energy detector triggered events
// found on file: hartlepool_1_pdf_data.root
//////////////////////////////////////////////////////////

#ifndef extract_h
#define extract_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class extract {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           gtid;
   Int_t           mcid;
   Int_t           subid;
   Int_t           inner_hit;
   Int_t           inner_hit_prev;
   Int_t           id_plus_dr_hit;
   Int_t           veto_hit;
   Int_t           veto_plus_dr_hit;
   Int_t           veto_hit_prev;
   Double_t        pe;
   Double_t        innerPE;
   Double_t        vetoPE;
   Double_t        n9;
   Double_t        n9_prev;
   Double_t        nOff;
   Double_t        n100;
   Double_t        n100_prev;
   Double_t        n400;
   Double_t        n400_prev;
   Double_t        nX;
   Double_t        nX_prev;
   Double_t        dn9prev;
   Double_t        dn100prev;
   Double_t        good_pos;
   Double_t        good_pos_prev;
   Double_t        good_dir;
   Double_t        good_dir_prev;
   Double_t        x;
   Double_t        y;
   Double_t        z;
   Double_t        t;
   Double_t        u;
   Double_t        v;
   Double_t        w;
   Double_t        azimuth_ks;
   Double_t        azimuth_ks_prev;
   Double_t        distpmt;
   Double_t        mc_energy;
   Double_t        mcx;
   Double_t        mcy;
   Double_t        mcz;
   Double_t        mct;
   Double_t        mcu;
   Double_t        mcv;
   Double_t        mcw;
   Double_t        closestPMT;
   Double_t        closestPMT_prev;
   Double_t        dxPrevx;
   Double_t        dyPrevy;
   Double_t        dzPrevz;
   Double_t        drPrevr;
   Double_t        drPrevrQFit;
   Double_t        dxmcx;
   Double_t        dymcy;
   Double_t        dzmcz;
   Double_t        drmcr;
   Double_t        dt_sub;
   Double_t        dt_prev_us;
   Double_t        timestamp;
   Int_t           num_tested;
   Double_t        best_like;
   Double_t        worst_like;
   Double_t        average_like;
   Double_t        average_like_05m;
   Double_t        beta_one;
   Double_t        beta_two;
   Double_t        beta_three;
   Double_t        beta_four;
   Double_t        beta_five;
   Double_t        beta_six;
   Double_t        beta_one_prev;
   Double_t        beta_two_prev;
   Double_t        beta_three_prev;
   Double_t        beta_four_prev;
   Double_t        beta_five_prev;
   Double_t        beta_six_prev;

   // List of branches
   TBranch        *b_gtid;   //!
   TBranch        *b_mcid;   //!
   TBranch        *b_subid;   //!
   TBranch        *b_inner_hit;   //!
   TBranch        *b_inner_hit_prev;   //!
   TBranch        *b_id_plus_dr_hit;   //!
   TBranch        *b_veto_hit;   //!
   TBranch        *b_veto_plus_dr_hit;   //!
   TBranch        *b_veto_hit_prev;   //!
   TBranch        *b_pe;   //!
   TBranch        *b_innerPE;   //!
   TBranch        *b_vetoPE;   //!
   TBranch        *b_n9;   //!
   TBranch        *b_n9_prev;   //!
   TBranch        *b_nOff;   //!
   TBranch        *b_n100;   //!
   TBranch        *b_n100_prev;   //!
   TBranch        *b_n400;   //!
   TBranch        *b_n400_prev;   //!
   TBranch        *b_nX;   //!
   TBranch        *b_nX_prev;   //!
   TBranch        *b_dn9prev;   //!
   TBranch        *b_dn100prev;   //!
   TBranch        *b_good_pos;   //!
   TBranch        *b_good_pos_prev;   //!
   TBranch        *b_good_dir;   //!
   TBranch        *b_good_dir_prev;   //!
   TBranch        *b_x;   //!
   TBranch        *b_y;   //!
   TBranch        *b_z;   //!
   TBranch        *b_t;   //!
   TBranch        *b_u;   //!
   TBranch        *b_v;   //!
   TBranch        *b_w;   //!
   TBranch        *b_azimuth_ks;   //!
   TBranch        *b_azimuth_ks_prev;   //!
   TBranch        *b_distpmt;   //!
   TBranch        *b_mc_energy;   //!
   TBranch        *b_mcx;   //!
   TBranch        *b_mcy;   //!
   TBranch        *b_mcz;   //!
   TBranch        *b_mct;   //!
   TBranch        *b_mcu;   //!
   TBranch        *b_mcv;   //!
   TBranch        *b_mcw;   //!
   TBranch        *b_closestPMT;   //!
   TBranch        *b_closestPMT_prev;   //!
   TBranch        *b_dxPrevx;   //!
   TBranch        *b_dyPrevy;   //!
   TBranch        *b_dzPrevz;   //!
   TBranch        *b_drPrevr;   //!
   TBranch        *b_drPrevrQFit;   //!
   TBranch        *b_dxmcx;   //!
   TBranch        *b_dymcy;   //!
   TBranch        *b_dzmcz;   //!
   TBranch        *b_drmcr;   //!
   TBranch        *b_dt_sub;   //!
   TBranch        *b_dt_prev_us;   //!
   TBranch        *b_timestamp;   //!
   TBranch        *b_num_tested;   //!
   TBranch        *b_best_like;   //!
   TBranch        *b_worst_like;   //!
   TBranch        *b_average_like;   //!
   TBranch        *b_average_like_05m;   //!
   TBranch        *b_beta_one;   //!
   TBranch        *b_beta_two;   //!
   TBranch        *b_beta_three;   //!
   TBranch        *b_beta_four;   //!
   TBranch        *b_beta_five;   //!
   TBranch        *b_beta_six;   //!
   TBranch        *b_beta_one_prev;   //!
   TBranch        *b_beta_two_prev;   //!
   TBranch        *b_beta_three_prev;   //!
   TBranch        *b_beta_four_prev;   //!
   TBranch        *b_beta_five_prev;   //!
   TBranch        *b_beta_six_prev;   //!

   extract(TTree *tree=0);
   virtual ~extract();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef extract_cxx
extract::extract(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("hartlepool_1_pdf_data.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("hartlepool_1_pdf_data.root");
      }
      f->GetObject("data",tree);

   }
   Init(tree);
}

extract::~extract()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t extract::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t extract::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void extract::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("gtid", &gtid, &b_gtid);
   fChain->SetBranchAddress("mcid", &mcid, &b_mcid);
   fChain->SetBranchAddress("subid", &subid, &b_subid);
   fChain->SetBranchAddress("inner_hit", &inner_hit, &b_inner_hit);
   fChain->SetBranchAddress("inner_hit_prev", &inner_hit_prev, &b_inner_hit_prev);
   fChain->SetBranchAddress("id_plus_dr_hit", &id_plus_dr_hit, &b_id_plus_dr_hit);
   fChain->SetBranchAddress("veto_hit", &veto_hit, &b_veto_hit);
   fChain->SetBranchAddress("veto_plus_dr_hit", &veto_plus_dr_hit, &b_veto_plus_dr_hit);
   fChain->SetBranchAddress("veto_hit_prev", &veto_hit_prev, &b_veto_hit_prev);
   fChain->SetBranchAddress("pe", &pe, &b_pe);
   fChain->SetBranchAddress("innerPE", &innerPE, &b_innerPE);
   fChain->SetBranchAddress("vetoPE", &vetoPE, &b_vetoPE);
   fChain->SetBranchAddress("n9", &n9, &b_n9);
   fChain->SetBranchAddress("n9_prev", &n9_prev, &b_n9_prev);
   fChain->SetBranchAddress("nOff", &nOff, &b_nOff);
   fChain->SetBranchAddress("n100", &n100, &b_n100);
   fChain->SetBranchAddress("n100_prev", &n100_prev, &b_n100_prev);
   fChain->SetBranchAddress("n400", &n400, &b_n400);
   fChain->SetBranchAddress("n400_prev", &n400_prev, &b_n400_prev);
   fChain->SetBranchAddress("nX", &nX, &b_nX);
   fChain->SetBranchAddress("nX_prev", &nX_prev, &b_nX_prev);
   fChain->SetBranchAddress("dn9prev", &dn9prev, &b_dn9prev);
   fChain->SetBranchAddress("dn100prev", &dn100prev, &b_dn100prev);
   fChain->SetBranchAddress("good_pos", &good_pos, &b_good_pos);
   fChain->SetBranchAddress("good_pos_prev", &good_pos_prev, &b_good_pos_prev);
   fChain->SetBranchAddress("good_dir", &good_dir, &b_good_dir);
   fChain->SetBranchAddress("good_dir_prev", &good_dir_prev, &b_good_dir_prev);
   fChain->SetBranchAddress("x", &x, &b_x);
   fChain->SetBranchAddress("y", &y, &b_y);
   fChain->SetBranchAddress("z", &z, &b_z);
   fChain->SetBranchAddress("t", &t, &b_t);
   fChain->SetBranchAddress("u", &u, &b_u);
   fChain->SetBranchAddress("v", &v, &b_v);
   fChain->SetBranchAddress("w", &w, &b_w);
   fChain->SetBranchAddress("azimuth_ks", &azimuth_ks, &b_azimuth_ks);
   fChain->SetBranchAddress("azimuth_ks_prev", &azimuth_ks_prev, &b_azimuth_ks_prev);
   fChain->SetBranchAddress("distpmt", &distpmt, &b_distpmt);
   fChain->SetBranchAddress("mc_energy", &mc_energy, &b_mc_energy);
   fChain->SetBranchAddress("mcx", &mcx, &b_mcx);
   fChain->SetBranchAddress("mcy", &mcy, &b_mcy);
   fChain->SetBranchAddress("mcz", &mcz, &b_mcz);
   fChain->SetBranchAddress("mct", &mct, &b_mct);
   fChain->SetBranchAddress("mcu", &mcu, &b_mcu);
   fChain->SetBranchAddress("mcv", &mcv, &b_mcv);
   fChain->SetBranchAddress("mcw", &mcw, &b_mcw);
   fChain->SetBranchAddress("closestPMT", &closestPMT, &b_closestPMT);
   fChain->SetBranchAddress("closestPMT_prev", &closestPMT_prev, &b_closestPMT_prev);
   fChain->SetBranchAddress("dxPrevx", &dxPrevx, &b_dxPrevx);
   fChain->SetBranchAddress("dyPrevy", &dyPrevy, &b_dyPrevy);
   fChain->SetBranchAddress("dzPrevz", &dzPrevz, &b_dzPrevz);
   fChain->SetBranchAddress("drPrevr", &drPrevr, &b_drPrevr);
   fChain->SetBranchAddress("drPrevrQFit", &drPrevrQFit, &b_drPrevrQFit);
   fChain->SetBranchAddress("dxmcx", &dxmcx, &b_dxmcx);
   fChain->SetBranchAddress("dymcy", &dymcy, &b_dymcy);
   fChain->SetBranchAddress("dzmcz", &dzmcz, &b_dzmcz);
   fChain->SetBranchAddress("drmcr", &drmcr, &b_drmcr);
   fChain->SetBranchAddress("dt_sub", &dt_sub, &b_dt_sub);
   fChain->SetBranchAddress("dt_prev_us", &dt_prev_us, &b_dt_prev_us);
   fChain->SetBranchAddress("timestamp", &timestamp, &b_timestamp);
   fChain->SetBranchAddress("num_tested", &num_tested, &b_num_tested);
   fChain->SetBranchAddress("best_like", &best_like, &b_best_like);
   fChain->SetBranchAddress("worst_like", &worst_like, &b_worst_like);
   fChain->SetBranchAddress("average_like", &average_like, &b_average_like);
   fChain->SetBranchAddress("average_like_05m", &average_like_05m, &b_average_like_05m);
   fChain->SetBranchAddress("beta_one", &beta_one, &b_beta_one);
   fChain->SetBranchAddress("beta_two", &beta_two, &b_beta_two);
   fChain->SetBranchAddress("beta_three", &beta_three, &b_beta_three);
   fChain->SetBranchAddress("beta_four", &beta_four, &b_beta_four);
   fChain->SetBranchAddress("beta_five", &beta_five, &b_beta_five);
   fChain->SetBranchAddress("beta_six", &beta_six, &b_beta_six);
   fChain->SetBranchAddress("beta_one_prev", &beta_one_prev, &b_beta_one_prev);
   fChain->SetBranchAddress("beta_two_prev", &beta_two_prev, &b_beta_two_prev);
   fChain->SetBranchAddress("beta_three_prev", &beta_three_prev, &b_beta_three_prev);
   fChain->SetBranchAddress("beta_four_prev", &beta_four_prev, &b_beta_four_prev);
   fChain->SetBranchAddress("beta_five_prev", &beta_five_prev, &b_beta_five_prev);
   fChain->SetBranchAddress("beta_six_prev", &beta_six_prev, &b_beta_six_prev);
   Notify();
}

Bool_t extract::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void extract::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t extract::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef extract_cxx
