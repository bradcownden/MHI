#ifndef QRFACTOR_HELPER
#define QRFACTOR_HELPER

#define SWITCH_CHAR				'-'

struct QRfactorOpts {
	char* sparse_mat_filename;	// by switch -matrix<filename>
	char* data_files;	// by switch -data<directory>
	int verbose;	// by switch -v<int> 
};


#endif // !QRFACTOR_HELPER