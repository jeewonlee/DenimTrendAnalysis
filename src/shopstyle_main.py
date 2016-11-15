import shopstyle_API
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def generate_data():
    ss = shopstyle_API.ShopStyle()
    popular_id=0
    #categories =['bootcut-jeans','classic-jeans','cropped-jeans','distressed-jeans','flare-jeans','relaxed-jeans','skinny-jeans','straight-leg-jeans','stretch-jeans']
    with open('../data/Shopstyle/jeans_data.csv','wb') as f:
        f.write('popular_id,location,product_id,currency,price,retailer_id,retailer_name, brand_id, brand_name,product_name,description,extract_date\n')
        for page in range(1488):
            data = ss.search(cat='jeans', sort='Popular',offset=page, limit=50)
            length = len(data['products'])
            for i in range(length):
                popular_id+=1
                if 'id' in data['products'][i]:
                    product_id = str(data['products'][i]['id'])
                if 'locale' in data['products'][i]:
                    location = str(data['products'][i]['locale'])
                if 'currency' in data['products'][i]:
                    currency = str(data['products'][i]['currency'])
                if 'retailer' in data['products'][i]:
                    if 'id' in data['products'][i]['retailer']:
                        retailer_ID = str(data['products'][i]['retailer']['id'])
                    if 'name' in data['products'][i]['retailer']:
                        retailer_name = str(data['products'][i]['retailer']['name']).replace(',','')
                if 'brand' in data['products'][i]:
                    if 'id' in data['products'][i]['brand']:
                        brand_id = str(data['products'][i]['brand']['id']).replace(',','')
                    if 'name' in data['products'][i]['brand']:
                        brand_name = str(data['products'][i]['brand']['name']).replace(',','')
                if 'price' in data['products'][i]:
                    price = str(data['products'][i]['price'])
                if 'name' in data['products'][i]:
                    name = str(data['products'][i]['name']).replace(',','')
                if 'description' in data['products'][i]:
                    description = str(data['products'][i]['description']).replace(',','')
                if 'extractDate' in data['products'][i]:
                    extract_date = str(data['products'][i]['extractDate'])
                f.write(str(popular_id)+','+location+','+product_id+','+currency+','+price+','+retailer_ID+','+retailer_name+','+brand_id+','+brand_name+','+name+','+description+','+extract_date+'\n')
    f.close()

def count_analysis():
    sumup = 0
    ss = shopstyle_API.ShopStyle()
    categories =['bootcut-jeans','classic-jeans','cropped-jeans','distressed-jeans','flare-jeans','relaxed-jeans','skinny-jeans','straight-leg-jeans','stretch-jeans']
    for c in categories:
        data = ss.search(cat=c, sort='Popular',offset=1, limit=2)
        print c+':\t'+str(data['metadata']['total'])
        sumup += data['metadata']['total']
    print 'sumup:\t', sumup
    data = ss.search(cat='jeans', sort='Popular',offset=1, limit=2)
    print 'total:\t'+str(data['metadata']['total'])

if __name__ == '__main__':
    ss = shopstyle_API.ShopStyle()
    data = ss.search(cat='jeans', sort='Popular',offset=1, limit=30)
    generate_data()
    #count_analysis()

    # bootcut-jeans	4075
    # classic-jeans	340
    # cropped-jeans	8100
    # distressed-jeans	5089
    # flare-jeans	4682
    # relaxed-jeans	3593
    # skinny-jeans	23041
    # straight-leg-jeans	19255
    # stretch-jeans	17510
    # total	74,427
    # sumup	85685
