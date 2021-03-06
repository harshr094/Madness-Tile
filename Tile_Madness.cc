#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include "legion.h"
#include <vector>
#include <queue>
#include <utility>

using namespace Legion;
using namespace std;



enum TASK_IDs{
    TOP_LEVEL_TASK_ID,
    REFINE_INTER_TASK_ID,
    REFINE_INTRA_TASK_ID,
    PRINT_TASK_ID,
    COMPRESS_INTRA_TASK_ID,
    COMPRESS_INTER_TASK_ID,
    RECONSTRUCT_INTER_TASK_ID,
    RECONSTRUCT_INTRA_TASK_ID,
    NORM_TASK_ID,
    INNER_PRODUCT_TASK_ID,
    GAXPY_INTER_TASK_ID,
    GAXPY_INTRA_TASK_ID,
};

enum FieldId{
    FID_X,
};

struct Arguments {
    int n;
    int l;
    int max_depth;
    coord_t idx;
    long int gen;
    Color partition_color;
    int actual_max_depth;
    int tile_height;
    Arguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _actual_max_depth=0, int _tile_height=1 )
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color), actual_max_depth(_actual_max_depth), tile_height(_tile_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct InnerProductArgs{
    int n;
    int l;
    int max_depth;
    coord_t idx;
    long int gen;
    Color partition_color1, partition_color2;
    int actual_max_depth;
    int tile_height;
    InnerProductArgs(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color1, Color _partition_color2, int _actual_max_depth=0, int _tile_height=1 )
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color1(_partition_color1), partition_color2(_partition_color2), actual_max_depth(_actual_max_depth), tile_height(_tile_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct GaxpyArgs{
    int n;
    int l;
    int max_depth;
    coord_t idx;
    long int gen;
    Color partition_color1, partition_color2, partition_color3;
    int pass;
    int actual_max_depth;
    int tile_height;
    bool left_null, right_null;
    GaxpyArgs(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color1, Color _partition_color2, Color _partition_color3, int _pass, bool _left_null, bool _right_null, int _actual_max_depth=0, int _tile_height=1 )
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color1(_partition_color1), partition_color2(_partition_color2), partition_color3(_partition_color3) ,pass(_pass), left_null(_left_null), right_null(_right_null), actual_max_depth(_actual_max_depth), tile_height(_tile_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct TreeArgs{
    int value;
    bool is_leaf;
    TreeArgs( int _value, bool _is_leaf ) : value(_value), is_leaf(_is_leaf) {}
};


struct HelperArgs{
    int level;
    coord_t idx;
    bool launch;
    int n;
    bool is_valid_entry;
    HelperArgs( int _level, coord_t _idx, bool _launch, int _n , bool _is_valid_entry ) : level(_level), idx(_idx), launch(_launch), n(_n), is_valid_entry( _is_valid_entry ) {}
};

struct GaxpyHelper{
    int n,l;
    coord_t idx;
    int pass;
    bool left_null, right_null;
    bool launch;
    GaxpyHelper( int _n, int _l, coord_t _idx, int _pass, bool _left_null, bool _right_null , bool _launch ) : n(_n), l(_l), idx(_idx), pass(_pass), left_null(_left_null), right_null(_right_null), launch(_launch)    {}
};

void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > read_acc(regions[0], FID_X);
    int node_counter=0;
    int max_depth = args.max_depth;
    queue<Arguments>tree;
    tree.push(args);
    while( !tree.empty() ){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        // if( n > max_depth )
        //     break;
        coord_t idx = temp.idx;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        node_counter++;
        cout<<node_counter<<": "<<n<<"~"<<l<<"~"<<idx<<"~"<<read_acc[idx].value<<endl;
        if(!read_acc[idx].is_leaf){
            Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, temp.tile_height);
            Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, temp.tile_height);
            tree.push( for_left_sub_tree );
            tree.push( for_right_sub_tree );
        }
    }
}

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    int overall_max_depth = 12;
    int actual_left_depth = 3;
    int tile_height = 4;

    long int seed = 12345;
    {
        const InputArgs &command_args = HighLevelRuntime::get_input_args();
        for (int idx = 1; idx < command_args.argc; ++idx)
        {
            if (strcmp(command_args.argv[idx], "-max_depth") == 0)
                overall_max_depth = atoi(command_args.argv[++idx]);
            else if (strcmp(command_args.argv[idx], "-seed") == 0)
                seed = atol(command_args.argv[++idx]);
            else if(strcmp(command_args.argv[idx],"--tile") == 0)
                tile_height = atoi( command_args.argv[++idx]);
        }
    }
    srand(time(NULL));
    Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)));
    IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }

    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;

    Arguments args1(0, 0, overall_max_depth, 0, partition_color1, actual_left_depth, tile_height);
    args1.gen = rand();
    cout<<"Launching Refine Task"<<endl;
    TaskLauncher refine_launcher(REFINE_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    refine_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    refine_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher);

    cout<<"Launching Print Task After Refine"<<endl;
    TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    RegionRequirement req3( lr1 , READ_ONLY, EXCLUSIVE, lr1 );
    req3.add_field(FID_X);
    print_launcher.add_region_requirement( req3 );
    runtime->execute_task(ctx, print_launcher);

    // cout<<"Launching Compress Task"<<endl;
    // TaskLauncher compress_launcher(COMPRESS_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // compress_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    // compress_launcher.add_field(0, FID_X);
    // runtime->execute_task(ctx, compress_launcher);

    // cout<<"Launching Print Task After Compress"<<endl;
    // TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // RegionRequirement req3( lr1 , READ_ONLY, EXCLUSIVE, lr1 );
    // req3.add_field(FID_X);
    // print_launcher.add_region_requirement( req3 );
    // runtime->execute_task(ctx, print_launcher);

    // cout<<"Launching Reconstruct Task"<<endl;
    // TaskLauncher reconstruct_launcher(RECONSTRUCT_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // reconstruct_launcher.add_region_requirement( RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1) );
    // reconstruct_launcher.add_field(0,FID_X);
    // runtime->execute_task(ctx,reconstruct_launcher);

    // cout<<"Launching Print Task After Reconstruct"<<endl;
    // runtime->execute_task(ctx, print_launcher);

    // cout<<"Launching Norm Task on Reconstruct Tree"<<endl;
    // TaskLauncher norm_launcher(NORM_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // norm_launcher.add_region_requirement( RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1) );
    // norm_launcher.add_field(0,FID_X);
    // Future f = runtime->execute_task( ctx, norm_launcher );
    // cout<<sqrt(f.get_result<int>())<<endl;

    Rect<1> tree_second(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)));
    IndexSpace is2 = runtime->create_index_space(ctx, tree_second);
    FieldSpace fs2 = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs2);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }
    LogicalRegion lr2 = runtime->create_logical_region(ctx, is2, fs2);
    Color partition_color2 = 20;
    Arguments args2(0, 0, overall_max_depth, 0, partition_color2, actual_left_depth, tile_height);
    args2.gen = rand();
    cout<<"Launching Refine Task For 2nd  Tree"<<endl;
    TaskLauncher refine_launcher2(REFINE_INTER_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    refine_launcher2.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    refine_launcher2.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher2);

    // cout<<"Launching Compress Task For 2nd Tree"<<endl;
    // TaskLauncher compress_launcher2(COMPRESS_INTER_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // compress_launcher2.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    // compress_launcher2.add_field(0, FID_X);
    // runtime->execute_task(ctx, compress_launcher2);

    cout<<"Launching Print Task For 2nd Tree"<<endl;
    TaskLauncher print_launcher2(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    RegionRequirement req4( lr2 , READ_ONLY, EXCLUSIVE, lr2 );
    req4.add_field(FID_X);
    print_launcher2.add_region_requirement( req4 );
    runtime->execute_task(ctx, print_launcher2);

    // cout<<"Launching Inner Product Task"<<endl;
    // InnerProductArgs args(0, 0, overall_max_depth, 0, partition_color1, partition_color2, actual_left_depth, tile_height);
    // TaskLauncher product_launcher(INNER_PRODUCT_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    // product_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // product_launcher.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2) );
    // product_launcher.add_field(0,FID_X);
    // product_launcher.add_field(1,FID_X);
    // Future result = runtime->execute_task( ctx, product_launcher );
    // cout<<result.get_result<int>()<<endl;

    Rect<1> gaxpy_tree(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)));
    IndexSpace isgaxpy = runtime->create_index_space(ctx, gaxpy_tree);
    FieldSpace fsgaxpy = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fsgaxpy);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }
    LogicalRegion lrgaxpy = runtime->create_logical_region(ctx, isgaxpy, fsgaxpy);
    Color partition_color3 = 30;
    GaxpyArgs args(0, 0, overall_max_depth, 0, partition_color1, partition_color2, partition_color3, 0, false, false, actual_left_depth, tile_height);
 
    cout<<"Launching Gaxpy Taks for Tree"<<endl;
    TaskLauncher gaxpy_launcher(GAXPY_INTER_TASK_ID, TaskArgument(&args, sizeof(GaxpyArgs)));
    RegionRequirement req1(lr1, READ_ONLY, EXCLUSIVE, lr1);
    req1.add_field(FID_X);
    RegionRequirement req2(lr2, READ_ONLY, EXCLUSIVE , lr2);
    req2.add_field(FID_X);
    RegionRequirement reqgaxpy(lrgaxpy, WRITE_DISCARD, EXCLUSIVE, lrgaxpy);
    reqgaxpy.add_field(FID_X);
    gaxpy_launcher.add_region_requirement(req1);
    gaxpy_launcher.add_region_requirement(req2);
    gaxpy_launcher.add_region_requirement(reqgaxpy);
    runtime->execute_task(ctx, gaxpy_launcher);
    cout<<"Launching Print Task for Gaxpy"<<endl;
    TaskLauncher print_gaxpy(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    RegionRequirement gaxpy_req( lrgaxpy , READ_ONLY, EXCLUSIVE, lrgaxpy );
    gaxpy_req.add_field(FID_X);
    print_gaxpy.add_region_requirement( gaxpy_req );
    runtime->execute_task(ctx, print_gaxpy );

}



void refine_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(regions[1], FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        long int node_value=rand();
        node_value = node_value % 10 + 1;
        if (node_value <= 3 || n == max_depth - 1) {
            tree_acc[idx].value = node_value % 3 + 1;
            tree_acc[idx].is_leaf =true;
        }
        else {
            tree_acc[idx].value = 0;
        }
        if( (node_value > 3 )&&( n < max_depth ) ){
            if( (n % tile_height )==( tile_height-1 ) ){
                helper_acc[helper_counter].level = l;
                helper_acc[helper_counter].idx = idx;
                helper_acc[helper_counter].n = n;
                helper_acc[helper_counter].launch = true;
                helper_counter++;
            }
            else{
                Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
        
    }
}


void gaxpy_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    GaxpyArgs args = task->is_index_space ? *(const GaxpyArgs *) task->local_args
    : *(const GaxpyArgs *) task->args;
    int tile_height = args.tile_height;
    queue<GaxpyArgs>tree;
    tree.push(args);
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    int helper_counter=0;
    int max_depth = args.max_depth;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree1(regions[0], FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree2(regions[1], FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree3(regions[2], FID_X);    
    const FieldAccessor<WRITE_DISCARD,GaxpyHelper,1,coord_t,Realm::AffineAccessor<GaxpyHelper,1,coord_t> > helper_acc(regions[3], FID_X);  

    while(!tree.empty()){
        GaxpyArgs temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        int pass = temp.pass;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        bool left_null = temp.left_null;
        bool right_null = temp.right_null;
        int value;
        if( n > max_depth )
            break;
        if( left_null ){
            if(tree2[idx].is_leaf){
                value = pass + tree2[idx].value;
                tree3[idx].value = value;
                tree3[idx].is_leaf = true;
            }
            else{
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = pass/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, max_depth, idx_left_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2, max_depth, idx_right_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }
            }
        }
        else if( right_null ){
            if( tree1[idx].is_leaf){
                value = pass + tree1[idx].value;
                tree3[idx].value = value;
                tree3[idx].is_leaf = true;
            }
            else{
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = pass/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, max_depth, idx_left_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2, max_depth, idx_right_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
        }
        else{
            if( (tree1[idx].is_leaf )&&( tree2[idx].is_leaf )){
                value = tree1[idx].value + tree2[idx].value;
                tree3[idx].value = value;
                tree3[idx].is_leaf = true;
            }
            else if(tree1[idx].is_leaf){
                value = tree1[idx].value;
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = value/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = true;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, max_depth, idx_left_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, true, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2, max_depth, idx_right_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, true, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
            else if(tree2[idx].is_leaf){
                value = tree2[idx].value;
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = value/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = true;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, max_depth, idx_left_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, left_null, true, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2, max_depth, idx_right_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, left_null, true, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
            else{
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = 0;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, max_depth, idx_left_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, 0, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2, max_depth, idx_right_sub_tree, temp.partition_color1, temp.partition_color2, temp.partition_color3, 0, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
        }
    }




}
void gaxpy_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    GaxpyArgs args = task->is_index_space ? *(const GaxpyArgs *) task->local_args
    : *(const GaxpyArgs *) task->args;
    int tile_height = args.tile_height;
    int max_depth = args.max_depth;
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(GaxpyHelper), FID_X);
    }
    LogicalRegion tree1 = regions[0].get_logical_region();
    LogicalRegion tree2 = regions[1].get_logical_region();
    LogicalRegion tree3 = regions[2].get_logical_region();
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    RegionRequirement req1(tree1, READ_ONLY, EXCLUSIVE, tree1);
    req1.add_field(FID_X);
    RegionRequirement req2(tree2, READ_ONLY, EXCLUSIVE, tree2);
    req2.add_field(FID_X);
    RegionRequirement req3(tree3, WRITE_DISCARD, EXCLUSIVE, tree3);
    req3.add_field(FID_X);
    RegionRequirement req4(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req4.add_field(FID_X);
    TaskLauncher gaxpy_intra_launcher(GAXPY_INTRA_TASK_ID, TaskArgument(&args,sizeof(GaxpyArgs)));
    gaxpy_intra_launcher.add_region_requirement(req1);
    gaxpy_intra_launcher.add_region_requirement(req2);
    gaxpy_intra_launcher.add_region_requirement(req3);
    gaxpy_intra_launcher.add_region_requirement(req4);
    runtime->execute_task(ctx,gaxpy_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req4 );
    const FieldAccessor<READ_ONLY,GaxpyHelper,1,coord_t,Realm::AffineAccessor<GaxpyHelper,1,coord_t> > read_acc(physicalRegion, FID_X);
    int task_counter = 0;
    vector<pair<coord_t,coord_t> >color_index;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++){
        if(!read_acc[i].launch)
            break;
        coord_t idx = read_acc[i].idx;
        int l = read_acc[i].l;
        int nx = read_acc[i].n;
        int pass = read_acc[i].pass;
        bool left_null = read_acc[i].left_null;
        bool right_null = read_acc[i].right_null;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, max_depth - nx));
        GaxpyArgs left_args( nx+1, 2*l , args.max_depth, idx_left_sub_tree, args.partition_color1, args.partition_color2, args.partition_color3, pass, left_null, right_null , args.actual_max_depth, args.tile_height);
        GaxpyArgs right_args( nx+1, 2*l +1 , args.max_depth, idx_right_sub_tree, args.partition_color1, args.partition_color2, args.partition_color3, pass, left_null, right_null , args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter, TaskArgument(&left_args, sizeof(GaxpyArgs)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(GaxpyArgs)));        
        task_counter++;
        color_index.push_back(make_pair(idx_left_sub_tree,idx_right_sub_tree-1));
        color_index.push_back(make_pair(idx_right_sub_tree, idx_right_sub_tree +  static_cast<coord_t>(pow(2, max_depth - nx)) - 2) );
    }
    if( task_counter > 0 ){
        IndexSpace is = tree3.get_index_space();
        DomainPointColoring coloring;
        for( int i = 0 ; i < task_counter ; i++ ){
            coloring[i]= Rect<1>(color_index[i].first,color_index[i].second);
        }
        Rect<1>color_space = Rect<1>(0,task_counter-1);
        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color3);
        LogicalPartition lp = runtime->get_logical_partition(ctx, tree3, ip);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher gaxpy_launcher(GAXPY_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        RegionRequirement newregion(lp,0,WRITE_DISCARD,EXCLUSIVE,tree3);
        newregion.add_field(0,FID_X);
        gaxpy_launcher.add_region_requirement(req1);
        gaxpy_launcher.add_region_requirement(req2);
        gaxpy_launcher.add_region_requirement(newregion);
        runtime->execute_index_space(ctx, gaxpy_launcher);
    }  
}

void refine_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    int max_depth = args.max_depth;
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher refine_intra_launcher(REFINE_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    refine_intra_launcher.add_region_requirement(req1);
    refine_intra_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,refine_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    int task_counter=0;
    vector<pair<coord_t,coord_t> >color_index;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        if(!read_acc[i].launch)
            break;
        coord_t idx = read_acc[i].idx;
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, max_depth - nx));
        Arguments left_args( nx+1 , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height);
        Arguments right_args( nx+1 , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
        color_index.push_back(make_pair(idx_left_sub_tree,idx_right_sub_tree-1));
        color_index.push_back(make_pair(idx_right_sub_tree, idx_right_sub_tree +  static_cast<coord_t>(pow(2, max_depth - nx)) - 2) );
    }
    if( task_counter > 0 ){
        IndexSpace is = lr.get_index_space();
        DomainPointColoring coloring;
        for( int i = 0 ; i < task_counter ; i++ ){
            coloring[i]= Rect<1>(color_index[i].first,color_index[i].second);
        }
        Rect<1>color_space = Rect<1>(0,task_counter-1);
        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
        LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher refine_launcher(REFINE_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        refine_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        refine_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, refine_launcher);
    }
}


void compress_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > read_acc(regions[0], FID_X);
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > write_acc(regions[1], FID_X);
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        write_acc[helper_counter].level = l;
        write_acc[helper_counter].idx = idx;
        write_acc[helper_counter].n = n;
        write_acc[helper_counter].is_valid_entry = true;
        if( ((n % tile_height ) ==( tile_height-1 ) && ( !read_acc[idx].is_leaf ) ) )
            write_acc[helper_counter].launch = true;
        else if( !read_acc[idx].is_leaf ){
                Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
        }
        helper_counter++;
    }
}


void compress_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height)-1));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher compress_intra_launcher(COMPRESS_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    RegionRequirement req1(lr, READ_ONLY, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    compress_intra_launcher.add_region_requirement( req1 );
    compress_intra_launcher.add_region_requirement( req2 );
    runtime->execute_task(ctx,compress_intra_launcher);
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    ArgumentMap arg_map;
    int task_counter=0;
    for( int  i = 0 ; i < (1<<tile_height) ; i++){
        bool launch = read_acc[i].launch;
        coord_t idx = read_acc[i].idx;
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        if( launch ){
            coord_t idx_left_sub_tree = idx+1;
            coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, args.max_depth - nx ));
            Arguments left_args( nx + 1 , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height);
            Arguments right_args( nx + 1 , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height);
            arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
            task_counter++;
            arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
            task_counter++;
        }
    }
    if( task_counter > 0 ){
        LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher compress_launcher(COMPRESS_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        compress_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        compress_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, compress_launcher);
    }
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > write_acc(regions[0], FID_X);
    for( int i = (1<<tile_height)-1; i>=0 ; i-- ){
        if( !read_acc[i].is_valid_entry )
            continue;
        coord_t idx = read_acc[i].idx;
        if( write_acc[idx].is_leaf )
            continue;
        int nx = read_acc[i].n;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, args.max_depth - nx));
        write_acc[idx].value = write_acc[idx_left_sub_tree].value + write_acc[idx_right_sub_tree].value;
    }
}

void reconstruct_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(regions[1], FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        if( tree_acc[idx].is_leaf )
            continue;
        int pass = tree_acc[idx].value/2;
        tree_acc[idx].value = 0;
        tree_acc[idx_left_sub_tree].value = tree_acc[idx_left_sub_tree].value + pass;
        tree_acc[idx_right_sub_tree].value = tree_acc[idx_right_sub_tree].value + pass;
        if( (n % tile_height )==( tile_height-1 ) ){
            helper_acc[helper_counter].n = n;
            helper_acc[helper_counter].idx = idx;
            helper_acc[helper_counter].level = l;
            helper_acc[helper_counter].launch = true;
            helper_counter++;
        }
        else{
            Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
            Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
            tree.push( for_left_sub_tree );
            tree.push( for_right_sub_tree );
        }
    }
}

void reconstruct_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    int max_depth = args.max_depth;
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher reconstruct_intra_launcher(RECONSTRUCT_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    reconstruct_intra_launcher.add_region_requirement(req1);
    reconstruct_intra_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,reconstruct_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    int task_counter=0;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        if(!read_acc[i].launch)
            break;
        coord_t idx = read_acc[i].idx;
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, max_depth - nx));
        Arguments left_args( nx+1 , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height);
        Arguments right_args( nx+1 , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
    }
    if( task_counter > 0 ){
        LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher reconstruct_launcher(RECONSTRUCT_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        reconstruct_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        reconstruct_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, reconstruct_launcher);
    }
}

int norm_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    int max_depth = args.max_depth;
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    RegionRequirement req(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req.add_field(FID_X);
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req );
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(physicalRegion, FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    int result=0;
    queue<Arguments>tree;
    tree.push(args);
    int helper_counter = 0;
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        result = result + tree_acc[idx].value*tree_acc[idx].value;
        if( tree_acc[idx].is_leaf )
            continue;
        if( (n% tile_height )==( tile_height - 1)){
            helper_acc[helper_counter].n = n;
            helper_acc[helper_counter].level =l;
            helper_acc[helper_counter].idx = idx;
            helper_acc[helper_counter].launch = true;
            helper_counter++;
        }
        else{
            Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
            Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
            tree.push( for_left_sub_tree );
            tree.push( for_right_sub_tree );
        }
    }
    int task_counter =0;
    ArgumentMap arg_map;
    for( int i = 0 ; i < helper_counter ; i++ ){
        coord_t idx = helper_acc[i].idx;
        int level = helper_acc[i].level;
        int nx = helper_acc[i].n;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, max_depth - nx));
        Arguments left_args( nx+1 , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height);
        Arguments right_args( nx+1 , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
    }
    if( task_counter > 0 ){
        LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher norm_launcher(NORM_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        norm_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        norm_launcher.add_field(0, FID_X);
        FutureMap f_result = runtime->execute_index_space(ctx, norm_launcher);
        for( int i = 0 ; i < task_counter ; i++ )
            result = result + f_result.get_result<int>(i);
    }
    return result;
}


int product_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    InnerProductArgs args = task->is_index_space ? *(const InnerProductArgs *) task->local_args
    : *(const InnerProductArgs *) task->args;
    int tile_height = args.tile_height;
    int max_depth = args.max_depth;
    LogicalRegion lr1 = regions[0].get_logical_region();
    LogicalRegion lr2 = regions[1].get_logical_region();
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    RegionRequirement req(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req.add_field(FID_X);
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req );
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(physicalRegion, FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree1(regions[0], FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree2(regions[1], FID_X);
    queue<InnerProductArgs>tree;
    tree.push(args);
    int result = 0;
    int helper_counter=0;
    while(!tree.empty()){
        InnerProductArgs temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        bool leaf1 = tree1[idx].is_leaf;
        bool leaf2 = tree2[idx].is_leaf;
        result = result + tree1[idx].value*tree2[idx].value;
        if(leaf1||leaf2)
            continue;
        if((n% tile_height )==( tile_height-1 )){
            helper_acc[helper_counter].n = n;
            helper_acc[helper_counter].level =l;
            helper_acc[helper_counter].idx = idx;
            helper_counter++;
        }
        else{
            InnerProductArgs for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color1, temp.partition_color2, temp.actual_max_depth, tile_height);
            InnerProductArgs for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color1, temp.partition_color2 ,temp.actual_max_depth, tile_height);
            tree.push( for_left_sub_tree );
            tree.push( for_right_sub_tree );
        }
    }
    int task_counter =0;
    ArgumentMap arg_map;
    for( int i = 0 ; i < helper_counter ; i++ ){
        coord_t idx = helper_acc[i].idx;
        int level = helper_acc[i].level;
        int nx = helper_acc[i].n;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, max_depth - nx));
        InnerProductArgs left_args( nx+1 , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color1 , args.partition_color2, args.actual_max_depth , args.tile_height);
        InnerProductArgs right_args( nx+1 , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color1, args.partition_color2 ,args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
    }
    if( task_counter > 0 ){
        LogicalPartition lp1 = runtime->get_logical_partition_by_color(ctx, lr1, args.partition_color1);
        LogicalPartition lp2 = runtime->get_logical_partition_by_color(ctx, lr2, args.partition_color2);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher product_launcher(INNER_PRODUCT_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        product_launcher.add_region_requirement(RegionRequirement(lp1,0,READ_ONLY, EXCLUSIVE, lr1));
        product_launcher.add_region_requirement(RegionRequirement(lp2,0,READ_ONLY, EXCLUSIVE, lr2));
        product_launcher.add_field(0,FID_X);
        product_launcher.add_field(1,FID_X);
        FutureMap f_result = runtime->execute_index_space(ctx, product_launcher);
        for( int i = 0 ; i < task_counter ; i++ )
            result = result + f_result.get_result<int>(i);
    }
    return result;
}

int main(int argc, char** argv){

    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(REFINE_INTER_TASK_ID, "refine_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<refine_inter_task>(registrar, "refine_inter");
    }

    {
        TaskVariantRegistrar registrar(REFINE_INTRA_TASK_ID, "refine_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<refine_intra_task>(registrar, "refine_intra");
    }

    {
        TaskVariantRegistrar registrar(PRINT_TASK_ID, "print");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<print_task>(registrar, "print");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_INTER_TASK_ID, "compress_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<compress_inter_task>(registrar, "compress_inter");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_INTRA_TASK_ID, "compress_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<compress_intra_task>(registrar, "compress_intra");
    }

    {
        TaskVariantRegistrar registrar(RECONSTRUCT_INTER_TASK_ID, "reconstruct_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<reconstruct_inter_task>(registrar, "reconstruct_inter");
    }

    {
        TaskVariantRegistrar registrar(RECONSTRUCT_INTRA_TASK_ID, "reconstruct_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<reconstruct_intra_task>(registrar, "reconstruct_intra");
    }

    {
        TaskVariantRegistrar registrar(NORM_TASK_ID, "norm_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<int,norm_task>(registrar, "norm_task");
    }

    {
        TaskVariantRegistrar registrar(INNER_PRODUCT_TASK_ID, "inner_product_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<int,product_task>(registrar, "inner_product_task");
    }

    {
        TaskVariantRegistrar registrar(GAXPY_INTRA_TASK_ID, "gaxpy_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<gaxpy_intra_task>(registrar, "gaxpy_intra");
    }

    {
        TaskVariantRegistrar registrar(GAXPY_INTER_TASK_ID, "gaxpy_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<gaxpy_inter_task>(registrar, "gaxpy_inter");
    }

    return Runtime::start(argc,argv);
}
