/*
 * bridges_gen.c: Minimal standalone Bridges puzzle generator
 * 
 * Usage:
 *   ./bridges_gen [count] [params]
 *   ./bridges_gen 10 7x7m2d0     # 10 easy 7x7 puzzles
 *   ./bridges_gen 1 10x10m2d2    # 1 hard 10x10 puzzle
 * 
 * Output format (one line per puzzle):
 *   PARAMS:DESC,SOLUTION
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "puzzles.h"

/* The game structure is defined in bridges.c */
extern const struct game thegame;

int main(int argc, char **argv)
{
    int n = 1;
    char *params_str = NULL;
    int i;
    
    /* Parse arguments */
    if (argc >= 2)
        n = atoi(argv[1]);
    if (n <= 0) n = 1;
    
    if (argc >= 3)
        params_str = argv[2];
    
    /* Generate puzzles */
    for (i = 0; i < n; i++) {
        char *aux_info = NULL;
        game_params *cur_params;
        char *desc;
        random_state *rs;
        char seed_buf[64];
        
        /* Create a random seed */
        sprintf(seed_buf, "%ld_%d", (long)time(NULL), i);
        rs = random_new(seed_buf, strlen(seed_buf));
        
        /* Get params */
        cur_params = thegame.default_params();
        if (params_str) {
            thegame.decode_params(cur_params, params_str);
        }
        
        /* Generate puzzle description and aux_info (solution) */
        desc = thegame.new_desc(cur_params, rs, &aux_info, false);
        
        /* Output */
        {
            char *enc_params = thegame.encode_params(cur_params, true);
            
            if (aux_info && aux_info[0] == 'S') {
                printf("%s:%s,%s\n", enc_params, desc, aux_info + 1);
            } else if (aux_info) {
                printf("%s:%s,%s\n", enc_params, desc, aux_info);
            } else {
                printf("%s:%s,\n", enc_params, desc);
            }
            
            sfree(enc_params);
        }
        
        sfree(desc);
        sfree(aux_info);
        thegame.free_params(cur_params);
        random_free(rs);
    }
    
    return 0;
}
