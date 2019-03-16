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
};

enum FieldId{
    FID_X,
};

struct Arguments {
    /* level of the node in the binary tree. Root is at level 0 */
    int n;

    /* labeling of the node in the binary tree. Root has the value label = 0 
    * Node with (n, l) has it's left child at (n + 1, 2 * l) and it's right child at (n + 1, 2 * l + 1)
    */
    int l;

    int max_depth;

    coord_t idx;

    long int gen;

    Color partition_color;

    int actual_max_depth;

    int tile_height;

    bool is_Root;

    Arguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _actual_max_depth=0, int _tile_height=1 , bool _is_root = false )
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color), actual_max_depth(_actual_max_depth), tile_height(_tile_height), is_Root(_is_root )
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }

};

struct HelperArgs
{
    int level;
    coord_t idx;
    HelperArgs( int _level, coord_t _idx ) : level(_level), idx(_idx) {}
};


void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    queue<Arguments>tree;
    tree.push(args);
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
        int node_value = read_acc[idx];
        cout<<"Print Node Value "<<n<<"~"<<l<<"~"<<idx<<"~"<<node_value<<endl;
        if( node_value == 0 ){
                Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height, false );
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height, false );
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
        }
    }
}



void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    int overall_max_depth = 7;
    int actual_left_depth = 4;
    int tile_height = 3;

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
    Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int), FID_X);
    }

    // For 1st logical region
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    // Any random value will work
    Color partition_color1 = 10;

    Arguments args1(0, 0, overall_max_depth, 0, partition_color1, actual_left_depth, tile_height,true);
    args1.gen = rand();
    // Launching the refine task
    TaskLauncher refine_launcher(REFINE_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    refine_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    refine_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher);
    TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    print_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    print_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, print_launcher);
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
        const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[0], FID_X);
        if (node_value <= 3 || n == max_depth - 1) {
            write_acc[idx] = node_value % 3 + 1;
        }
        else {
            write_acc[idx] = 0;
        }
        if( (node_value > 3 )&&( n < max_depth ) ){
            if( (n % tile_height )==( tile_height-1 ) ){
                const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > write_acc(regions[1], FID_X);
                write_acc[helper_counter].level = l;
                write_acc[helper_counter].idx = idx;
                helper_counter++;
            }
            else{
                Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height, false );
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height, false );
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
        
    }

}

void refine_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    bool is_Root = args.is_Root;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    int n = args.n;
    if(is_Root){
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
        args.is_Root = false;
        TaskLauncher refine_launcher(REFINE_INTER_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
        RegionRequirement req3(lr,WRITE_DISCARD,EXCLUSIVE,lr);
        RegionRequirement req4(new_helper_Region,READ_ONLY,EXCLUSIVE,new_helper_Region);
        req3.add_field(FID_X);
        req4.add_field(FID_X);
        refine_launcher.add_region_requirement(req3);
        refine_launcher.add_region_requirement(req4);
        runtime->execute_task(ctx, refine_launcher);
    }
    else{
        cout<<"Helper Region "<<n+tile_height<<"~"<<args.l<<"~"<<args.idx<<endl;
        ArgumentMap arg_map;
        const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(regions[1], FID_X);
        int task_counter=0;
        vector<pair<coord_t,coord_t> >color_index;
        coord_t right_idx_add = pow(2, args.max_depth - (n + tile_height -1 ) );
        for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
            coord_t idx = read_acc[i].idx;
            int level = read_acc[i].level;
            if( idx == 0 )
                break;
            coord_t idx_left_sub_tree = idx+1;
            coord_t idx_right_sub_tree = idx+right_idx_add;
            Arguments left_args( n + tile_height , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height , true );
            Arguments right_args( n + tile_height , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height, true );
            arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
            task_counter++;
            arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
            task_counter++;
            color_index.push_back(make_pair(idx_left_sub_tree,idx_right_sub_tree-1));
            color_index.push_back(make_pair(idx_right_sub_tree, idx_right_sub_tree + right_idx_add - 2));
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

    return Runtime::start(argc,argv);
}
