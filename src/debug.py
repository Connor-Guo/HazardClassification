class Loader(object):
    def __init__(self):
        pass

    def _l(self, fp: str):
        with open(fp, encoding='UTF-8') as f:
            for line in f.readlines():
                if line:
                    if line.startswith('#'):  # 忽略注释行
                        continue
                    else:
                        yield line.strip()

    def load(self, fp: str):
        return list(self._l(fp))


class GibberishLoader(Loader):  # 读取乱码列表
    def load(self, fp: str):
        l = list(self._l(fp))
        l.extend(['\t', '\uf06c'])
        return l


class SymLoader(Loader):  # 读取近义词词典
    def load(self, fp: str):
        lines = list(self._l(fp))
        sym_dict = {}
        for line in lines:
            if line[0].isdigit():
                term_no = int(line.split(' ')[0])
                print(f"Symloader: {line} loaded.")
            else:
                list_terms = line.split('/')
                for subterm in list_terms:
                    sym_dict[subterm] = term_no
        return sym_dict


SymLoader().load('../data/sym_dict/unsafe_管制.txt')
