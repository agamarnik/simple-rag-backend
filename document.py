class Document:

    def __init__(self, title, content):
        self.title = title
        self.content = content

    def word_count(self): # returns the number of words in content
        words = self.content.split()  # Splits the string by whitespace into a list of words
        count = len(words)
        return count

    def preview(self, n): #returns the first n characters of content
        first_n_chars = self.content[:n]
        return first_n_chars

# testing
doc = Document("Test Title", "This is some sample content here")
print(doc.word_count())     # should print 6
print(doc.preview(10))      # should print This is so
