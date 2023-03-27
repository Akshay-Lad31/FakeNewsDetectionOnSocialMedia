from wordcloud import WordCloud
import matplotlib.pyplot as plt

def word_cloud_plot(words, title):
    plt.figure(figsize = (12,16)) 
    wordcloud = WordCloud(min_font_size = 3,  max_words = 1000 , width = 1200 , height = 800, collocations=False).generate(" ".join(words))
    plt.imshow(wordcloud,interpolation = 'bilinear')
    plt.title(title)

def plot_histogram(fake_news, real_news,axis, plot_title = "", x_label = ""):
    plt.figure(figsize = (12,16))
    plt.axis(axis)
    plt.hist(fake_news, bins=range(min(fake_news), max(fake_news) + 1, 1), 
              alpha=0.4, color="red")
    
    plt.hist(real_news, bins=range(min(real_news), max(real_news) + 1, 1),
                  alpha=0.4, color="blue")
    labels = ['Fake',"Real"]
    plt.legend(labels)
    plt.xlabel(x_label)
    plt.ylabel("Proportion")
    plt.title(plot_title)