from conans import ConanFile

class ConsistentSetConan(ConanFile):
    name = 'consistent_set'
    version = '0.1'
    exports_sources = 'include/*'
    no_copy_source = True

    def package(self):
        self.copy('*.h')

    def configure(self):
        if self.settings.compiler.cppstd in [None, '11', '14']:
            raise ConanInvalidConfiguration('C++ standard less than 17 not supported')