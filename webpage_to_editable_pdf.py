import asyncio
from pyppeteer import launch
from typing import Optional, Dict

async def _capture_pdf(url: str, output_path: str, options: Optional[Dict] = None) -> None:
    """
    异步函数: 使用Chrome 捕获网页为 PDF
    """
    # 默认配置(可覆盖)
    default_options = {
        'path': output_path,         
        'format': 'A4',               
        'margin': {              
            'top': '0.75',
            'right': '0.75',
            'bottom': '0.75',
            'left': '0.75'
        },
        'printBackground': True,      
        'waitUntil': 'networkidle2', 
        'slowMo': 100                
    }
    if options:
        default_options.update(options)  

    browser = await launch(
        headless=True,              
        args=[
            '--no-sandbox',           
            '--disable-blink-features=AutomationControlled' 
        ],
        ignoreHTTPSErrors=True      
    )
    
    try:

        page = await browser.newPage()
        

        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        await page.setUserAgent(user_agent)
        
        
        await page.evaluateOnNewDocument('''() => {
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined 
            })
        }''')
        
        await page.goto(url, {'waitUntil': default_options['waitUntil']})
        
        await page.setViewport({'width': 1280, 'height': 800})
        
        await page.pdf(default_options)
        print(f"PDF 生成成功！路径: {output_path}")
        
    except Exception as e:
        print(f" 生成失败: {str(e)}")
    finally:
        await browser.close()  

def webpage_to_editable_pdf(
    url: str,
    output_path: str = "output.pdf",
    options: Optional[Dict] = None
) -> None:
    """
    同步接口：将网页转换为可编辑 PDF(封装异步函数)
    
    参数:
        url (str): 目标网页 URL(支持 HTTP/HTTPS、本地文件 file:/// 路径)
        output_path (str): 输出 PDF 路径(默认: output.pdf)
        options (dict): 自定义配置(可选，覆盖默认值)
    """
    asyncio.get_event_loop().run_until_complete(
        _capture_pdf(url, output_path, options)
    )


if __name__ == "__main__":

    webpage_to_editable_pdf(
        url="https://mp.weixin.qq.com/",
        output_path="wechat_article.pdf",
        options={
            'waitUntil': 'networkidle0',  
            'margin': {'top': '0.5in'} 
        }
    )
    
    # 转换本地 HTML 文件
    # local_html_path = "file:///Users/zzliu/Desktop/test.html"  
    # webpage_to_editable_pdf(
    #     url=local_html_path,
    #     output_path="local_page.pdf",
    #     options={
    #         'format': 'Letter',          
    #         'printBackground': False     
    #     }
    # )