#include "dpdfnet.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"CPU contract line %d\n",__LINE__);return 1;}}while(0)
int main(int argc,char **argv) {
    CHECK(argc==2);
    int expected=atoi(argv[1]);CHECK(expected==0||expected==1);
    CHECK(dpdfnet_cpu_supported()==expected);
    for(int model=2;model<=8;model+=6){
        const dpdfnet_model_info *info=dpdfnet_get_model_info((dpdfnet_model_id)model);CHECK(info);
        float *state=malloc(info->state_floats*sizeof(float));CHECK(state);
        CHECK(!dpdfnet_init_state((dpdfnet_model_id)model,state));CHECK(state[0]!=0);
        if(!expected){
            float *weights=calloc(info->weight_floats,sizeof(float));CHECK(weights);
            CHECK(!dpdfnet_create((dpdfnet_model_id)model,weights,info->weight_floats));free(weights);
        }
        free(state);
    }
    printf("{\"cpu_supported\":%d,\"metadata_and_state_init\":true,\"unsupported_create_rejected\":%s}\n",expected,expected?"null":"true");
    return 0;
}
