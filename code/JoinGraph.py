import Utils
import copy

class JoinNode:
    def __init__(self, left, left_col, right, right_col, jid):
        self.left_col = left_col
        self.right_col = right_col
        self.left = left
        self.right = right
        self.jid = jid

    def generateTableEstimateKernel(self, f, query, estimator, stats):
        self.left.generateTableEstimateKernel(f, query, estimator, stats)
        self.right.generateTableEstimateKernel(f, query, estimator, stats)

    def generateJoinEstimateKernelBottomUp(self,f,query,estimator):
        self.left.generateJoinEstimateKernelBottomUp(f,query,estimator)
        self.right.generateJoinEstimateKernelBottomUp(f,query,estimator)

        icols = Utils.generateInvariantColumns(query)
        jcols = Utils.generateJoinColumns(query)

        t1, c1 = self.left_col
        t2, c2 = self.right_col
        #Are we the root join?
        if estimator.estimator == "AGPUJKDE" or estimator.estimator == "AGPUJKDE_COUNT":
            if isinstance(self.left, SampleScan):
                #Load all required values
                for c in jcols[t1]:
                    print("    unsigned int val_t%s_c%s = t%s_c%s[offset+get_global_id(0)];" % (t1,c,t1,c), file=f)
                if icols[t1]:
                    print("    double c_t%s = inv_t%s[offset+get_global_id(0)];" % (t1,t1), file=f)

            #And now, we need to perform the binary search.
            if estimator.join_kernel == "Cat":
                print("%sunsigned int pos_t%s = binarySearch(t%s_c%s, val_t%s_c%s - limit_t%s_c%s_t%s_c%s, n_t%s);" % ("    "*self.jid,t2,t2,c2,t1,c1,t1,c1,t2,c2,t2), file=f)
                print("%swhile(pos_t%s < n_t%s && t%s_c%s[pos_t%s] == (unsigned int) (val_t%s_c%s+limit_t%s_c%s_t%s_c%s)){" % ("    "*self.jid,t2,t2,t2,c2,t2,t1,c1,t1,c1,t2,c2), file=f)
            else:
                print("%sunsigned int pos_t%s = binarySearch(t%s_c%s, val_t%s_c%s - limit_t%s_c%s_t%s_c%s, n_t%s);" % (
                "    " * self.jid, t2, t2, c2, t1, c1, t1, c1, t2, c2, t2), file=f)
                print("%swhile(pos_t%s < n_t%s && t%s_c%s[pos_t%s] <= val_t%s_c%s+limit_t%s_c%s_t%s_c%s){" % (
                "    " * self.jid, t2, t2, t2, c2, t2, t1, c1, t1, c1, t2, c2), file=f)

            #Load contributions and variables

            for col in jcols[t2]:
                print("%sunsigned int val_t%s_c%s = t%s_c%s[pos_t%s];" % ("    "*(self.jid+1),t2,col,t2,col,t2), file=f)
            if icols[t2]:
                print("%sdouble c_t%s = inv_t%s[pos_t%s];" % ("    "*(self.jid+1),t2,t2,t2), file=f)
        elif estimator.estimator == "GPUSample" or estimator.estimator == "GPUCorrelatedSample":
            if isinstance(self.left, SampleScan):
                #Load all required values
                for c in jcols[t1]:
                    print("    unsigned int val_t%s_c%s = t%s_c%s[offset+get_global_id(0)];" % (t1,c,t1,c), file=f)

            #And now, we need to perform the binary search.
            print("%sunsigned int pos_t%s = binarySearch(t%s_c%s, val_t%s_c%s, n_t%s);" % ("    "*self.jid,t2,t2,c2,t1,c1,t2), file=f)
            print("%swhile(pos_t%s < n_t%s && t%s_c%s[pos_t%s] == val_t%s_c%s){" % ("    "*self.jid,t2,t2,t2,c2,t2,t1,c1), file=f)
            for col in jcols[t2]:
                print("%sunsigned int val_t%s_c%s = t%s_c%s[pos_t%s];" % ("    "*(self.jid+1),t2,col,t2,col,t2), file=f)
        else:
            raise Exception("This estimator is not handled by the join graph.")

    def getSortColumnDict(self,dict):
        if isinstance(self.left, SampleScan):
            dict[self.left_col[0]] = self.left_col[1]
        dict[self.right_col[0]] = self.right_col[1]
        self.left.getSortColumnDict(dict)
        self.right.getSortColumnDict(dict)

    def generateJoinEstimateKernelTopDown(self,f,query):
        t1, c1 = self.left_col
        t2, c2 = self.right_col
        print("%spos_t%s++;" % ("    "*(self.jid+1),t2), file=f)
        print("%s}" % ("    "*self.jid), file=f)
        self.left.generateJoinEstimateKernelTopDown(f,query)
        self.right.generateJoinEstimateKernelTopDown(f,query)

    def generateTableCode(self, f, query, estimator, limit=0, cu_factor = 2048):
        # Joins have no business in table code generation
        self.left.generateTableCode(f, query, estimator, limit, cu_factor)
        self.right.generateTableCode(f, query, estimator, limit, cu_factor)
        return self

    def collectTableIDs(self):
        tids = []
        tids += self.left.collectTableIDs()
        tids += self.right.collectTableIDs()
        return tids

    def collectJoinPairs(self):
        t1, c1 = self.left_col
        t2, c2 = self.right_col
        tids = []
        tids += self.left.collectJoinPairs()
        tids += self.right.collectJoinPairs()
        return tids + [(t1,c1,t2,c2)]

    def __str__(self):
        return "join(%s = %s, %s, %s)" % (self.left_col, self.right_col, self.left, self.right)


class SampleScan:
    def __init__(self, table):
        self.table = table

    def getSortColumnDict(self, dict):
        return dict

    def collectJoinPairs(self):
        return []

    def generateJoinEstimateKernelTopDown(self,f,query):
        pass

    def generateJoinEstimateKernelBottomUp(self,f,query,estimator):
        pass

    def generateTableEstimateKernel(self, f, query, estimator, stats):
        ts, dvals = stats
        cols = Utils.generateInvariantColumns(query)

        if estimator.estimator == "AGPUJKDE" or estimator.estimator == "AGPUJKDE_COUNT":
            kernels = [estimator.kernels[self.table][index] for index in cols[self.table]]
            if len(kernels) != 0:
                print("__kernel void invk_t%s(" % self.table, file=f)
                for i, k in enumerate(kernels):
                    if k == "GaussRange":
                        print("    __global unsigned int* c%s, T h%s, unsigned int  u%s, unsigned int l%s," % (i, i, i, i), file=f)
                    elif k == "GaussPoint" or k == "CategoricalPoint":
                        print("    __global unsigned int* c%s, T h%s, unsigned int p%s," % (i, i, i), file=f)
                    else:
                        raise Exception("Unsupported kernel.")
                print("    __global T* o, unsigned int ss){", file=f)
                print("    for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){", file=f)
                print("        if(get_global_id(0)+offset < ss){", file=f)
                for i, k in enumerate(kernels):
                    if k == "GaussPoint":
                        print("            T ec%s = gaussPointEst(c%s[offset+get_global_id(0)], h%s, p%s);" % (i, i, i, i), file=f)
                    elif k == "GaussRange":
                        print("            T ec%s = gaussRangeEst(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (
                        i, i, i, i, i), file=f)
                    elif k == "CategoricalPoint":
                        print("            T ec%s = catPointEst(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (
                        i, i, i, i, dvals[self.table][i]), file=f)
                    else:
                        raise Exception("Unsupported kernel.")
                print("            o[offset+get_global_id(0)] = 1.0 ", end=' ', file=f)
                for i, k in enumerate(kernels):
                    print("* ec%s" % i, end=' ', file=f)
                print(";", file=f)
                print("       }", file=f)
                print("   }", file=f)
                print("}", file=f)
                print(file=f)

        elif estimator.estimator == "GPUSample" or estimator.estimator == "GPUCorrelatedSample":
            if len(cols[self.table]) != 0:
                print("__kernel void invk_t%s(" % self.table, file=f)
                for i,j in enumerate(cols[self.table]):
                    if query.tables[self.table].columns[j].type == "range":
                        print("    __global unsigned int* c%s, unsigned int  u%s, unsigned int l%s," % (i, i, i), file=f)
                    elif query.tables[self.table].columns[j].type == "point":
                        print("    __global unsigned int* c%s, unsigned int p%s," % (i, i), file=f)
                    else:
                        raise Exception("Unsupported ctype.")
                print("    __global int* o, unsigned int ss){", file=f)
                print("    for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){", file=f)
                print("        if(get_global_id(0)+offset < ss) o[offset + get_global_id(0)+1] = 1 ", end=' ', file=f)
                for i,j in enumerate(cols[self.table]):
                    if query.tables[self.table].columns[j].type == "range":
                        print(" && c%s[offset + get_global_id(0)] != 0 && c%s[offset + get_global_id(0)] >= l%s && c%s[offset + get_global_id(0)] <= u%s " % (i,i, i, i,i), end=' ', file=f)
                    elif query.tables[self.table].columns[j].type == "point":
                        print(" && c%s[offset+get_global_id(0)] == p%s " % (i, i), end=' ', file=f)
                    else:
                        raise Exception("Unsupported ctype.")
                print(";", file=f)
                print("    }", file=f)
                print("}", file=f)
                print(file=f)
        else:
            raise Exception("Join graph does not support this estimator.")

    def collectTableIDs(self):
        return [self.table]

    def generateTableCode(self, f, query, estimator, limit=0, cu_factor=2048):
        cols = Utils.generateInvariantColumns(query)
        jcols = Utils.generateJoinColumns(query)
        i = self.table
        indices = cols[i]
        jindices = jcols[i]

        if estimator.estimator == "AGPUJKDE" or estimator.estimator == "AGPUJKDE_COUNT":
            # Compute invariant contributions and return qualifying tuple ids.
            if len(indices) != 0:
                print("    size_t local%s = 64;" % i, file=f)
                print("    size_t global%s = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((p->ss%s-1)/local%s+1)*local%s);" % (i,cu_factor,i,i,i), file=f)
                print("    p->invk%s.set_args(" % i, end=' ', file=f)
                for j in indices:
                    if estimator.kernels[i][j] == "GaussRange":
                        print("p->s_t%s_c%s, p->bw_t%s_c%s, u_t%s_c%s, l_t%s_c%s," % (i, j, i, j, i, j, i, j), end=' ', file=f)
                    else:
                        print("p->s_t%s_c%s, p->bw_t%s_c%s, p_t%s_c%s," % (i, j, i, j, i, j), end=' ', file=f)
                print(" p->inv_t%s, (unsigned int) p->ss%s" % (i,i), end=' ', file=f)
                print(");", file=f)
                print("    boost::compute::event ev%s = p->queue.enqueue_nd_range_kernel(p->invk%s,1,NULL,&(global%s), &(local%s));" % (i, i, i, i), file=f)
                # print >>f, "    ev%s.wait();" % i

                print("    boost::compute::transform(p->inv_t%s.begin(), p->inv_t%s.end(), p->count_t%s.begin()+1, boost::compute::lambda::_1 > %e, p->queue);" % (i, i, i, limit), file=f)
                print("    boost::compute::inclusive_scan(p->count_t%s.begin(), p->count_t%s.end(), p->map_t%s.begin(), p->queue);" % (i, i, i), file=f)
                print("    p->queue.finish();", file=f)
                print("    size_t rss_t%s = p->map_t%s[p->ss%s]+1;" % (i, i, i), file=f)
                print("    if(rss_t%s == 0) return 0.0;" % (i), file=f)
                for j in jindices:
                    print("    boost::compute::scatter_if(p->s_t%s_c%s.begin(), p->s_t%s_c%s.end(), p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->sr_t%s_c%s.begin(), p->queue);" % (i,j,i,j,i,i,i,j), file=f)
                print("    boost::compute::scatter_if(p->inv_t%s.begin(), p->inv_t%s.end(),p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->invr_t%s.begin(), p->queue);" % (i,i,i,i,i), file=f)
            else:
                # If there is no selection on this table, we don't have to do anything.
                print("    size_t rss_t%s = p->ss%s;" % (self.table,self.table), file=f)
            print(file=f)

        elif estimator.estimator == "GPUSample" or estimator.estimator == "GPUCorrelatedSample":
            # Compute invariant contributions and return qualifying tuple ids.
            if len(indices) != 0:
                print("    size_t local%s = 64;" % i, file=f)
                print("    size_t global%s = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((p->ss%s-1)/local%s+1)*local%s);" % (i,cu_factor,i,i,i), file=f)
                print("    p->invk%s.set_args(" % i, end=' ', file=f)
                for j in indices:
                    if query.tables[i].columns[j].type == "range":
                        print("p->s_t%s_c%s, u_t%s_c%s, l_t%s_c%s," % (i, j, i, j, i, j), end=' ', file=f)
                    elif query.tables[i].columns[j].type == "point":
                        print("p->s_t%s_c%s, p_t%s_c%s," % (i, j, i, j), end=' ', file=f)
                    else:
                        raise Exception("Unsupported type.")
                print(" p->count_t%s, (unsigned int) p->ss%s" % (i,i), end=' ', file=f)
                print(");", file=f)
                print("    boost::compute::event ev%s = p->queue.enqueue_nd_range_kernel(p->invk%s,1,NULL,&global%s, &local%s);" % (i, i,i,i), file=f)
                # print >>f, "    ev%s.wait();" % i

                #print >> f, "    boost::compute::transform(p->inv_t%s.begin(), p->inv_t%s.end(), p->count_t%s.begin()+1, boost::compute::lambda::_1 > %e, p->queue);" % (i, i, i, limit)
                print("    boost::compute::inclusive_scan(p->count_t%s.begin(), p->count_t%s.end(), p->map_t%s.begin(), p->queue);" % (i, i, i), file=f)
                print("    p->queue.finish();", file=f)
                print("    size_t rss_t%s = p->map_t%s[p->ss%s]+1;" % (i, i, i), file=f)
                print("    if(rss_t%s == 0) return 0.0;" % (i), file=f)
                for j in jindices:
                    print("    boost::compute::scatter_if(p->s_t%s_c%s.begin(), p->s_t%s_c%s.end(), p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->sr_t%s_c%s.begin(), p->queue);" % (i,j,i,j,i,i,i,j), file=f)
                print("    boost::compute::scatter_if(p->inv_t%s.begin(), p->inv_t%s.end(),p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->invr_t%s.begin(), p->queue);" % (i,i,i,i,i), file=f)
            else:
                # If there is no selection on this table, we don't have to do anything.
                print("    size_t rss_t%s = p->ss%s;" % (self.table,self.table), file=f)
            print(file=f)
        else:
            raise Exception("The join graph does not handle this type of estimator.")


    def generateJoinCode(self, f, estimator, ts):
        # Sample scans have no business in join code generation
        return self

    def __str__(self):
        return "SS(%s)" % str(self.table)


def constructJoinGraph(query):
    tables = [set() for _ in query.joins]

    joins = copy.deepcopy(query.joins)

    for i, join in enumerate(query.joins):
        for j in join:
            tables[i].add(j[0])
    # Start off with the initial join
    jc = tuple(joins[0][0])
    tree = SampleScan(jc[0])

    jid = 1
    for j in joins[0][1:]:
        tree = JoinNode(tree, jc, SampleScan(j[0]), tuple(j), jid)
        jid += 1
    tree_tables = tables.pop(0)
    joins.pop(0)

    while joins:
        for i, (join, table) in enumerate(zip(joins, tables)):
            # Okay, this join works together with our previous join tree.
            if tree_tables & table:
                join = copy.deepcopy(join)
                col_left = None
                for k, tcol in enumerate(join):
                    x, y = tcol
                    # We found the column from the other join.
                    if x in tree_tables:
                        col_left = (x, y)
                        join.pop(k)
                        break
                if col_left is None:
                    raise Exception("Join tree construction failed. Does your query contain a cross?")

                jc = None
                for tcol in join:
                    x, y = tcol
                    tree = JoinNode(tree, col_left, SampleScan(x), (x, y), jid)
                    jid += 1

                tree_tables |= tables.pop(i)
                joins.pop(i)
                break
    return tree
