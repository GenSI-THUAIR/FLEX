from openai import OpenAI
import time
import os

def test_api_connection():
    """测试API连接和基本功能"""
    client = OpenAI(
        base_url=os.getenv("BASE_URL", ""),
        api_key=os.getenv("API_KEY", "")
    )

    print("🔗 测试API连接...")
    
    try:
        # 测试简单的数学问题
        prompt = "Solve the equation: x^2-2x+1=0. Show your work step by step."
        
        print(f"📝 发送测试问题: {prompt}")
        print(f"测试模型：Qwen/Qwen2.5-32B-Instruct")
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct",  
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        end_time = time.time()
        
        print(f"✅ API调用成功!")
        print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
        print(f"📄 响应内容:")
        print("-" * 50)
        print(response.choices[0].message.content)
        print("-" * 50)
        
        # 显示token使用情况
        if hasattr(response, 'usage') and response.usage:
            print(f"📊 Token使用情况:")
            print(f"   输入tokens: {response.usage.prompt_tokens}")
            print(f"   输出tokens: {response.usage.completion_tokens}")
            print(f"   总tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")
        return False

def test_multiple_models():
    """测试多个模型"""
    client = OpenAI(
        base_url="http://35.220.164.252:3888/v1/",
        api_key="sk-MjWWt8xBcWvlcVRn7VoMAPSck5zOBbmrR2LXpWcBnBttXeRw"
    )
    
    # 从main.py中提取的模型列表
    models_to_test = [
        "claude-3-7-sonnet-latest",
        "gemini-2.5-pro",
        "gpt-4"
    ]
    
    test_prompt = "What is 15 + 27?"
    
    print("\n🧪 测试多个模型...")
    
    for model in models_to_test:
        print(f"\n📱 测试模型: {model}")
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            end_time = time.time()
            
            print(f"   ✅ 成功 (耗时: {end_time - start_time:.2f}秒)")
            print(f"   💬 回答: {response.choices[0].message.content[:100]}...")
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)}")

def test_math_problem():
    """测试复杂数学问题（类似DAPO-MATH数据集）"""
    client = OpenAI(
        base_url="http://35.220.164.252:3888/v1/",
        api_key="sk-MjWWt8xBcWvlcVRn7VoMAPSck5zOBbmrR2LXpWcBnBttXeRw"
    )
    
    # 类似于DAPO-MATH的复杂问题
    complex_prompt = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

In triangle ABC, sin∠A = 4/5 and ∠A < 90°. Let D be a point outside triangle ABC such that ∠BAD = ∠DAC and ∠BDC = 90°. Suppose that AD = 1 and that BD/CD = 3/2. If AB + AC can be expressed in the form a√b/c where a, b, c are pairwise relatively prime integers, find a + b + c.

Remember to put your answer on its own line after "Answer:"."""

    print("\n🔢 测试复杂数学问题...")
    print(f"📝 问题预览: {complex_prompt[:100]}...")
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="claude-3-7-sonnet-latest",
            messages=[{"role": "user", "content": complex_prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        end_time = time.time()
        
        print(f"✅ 复杂问题求解成功!")
        print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
        print(f"📄 完整响应:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
        # 尝试提取答案
        content = response.choices[0].message.content
        if "Answer:" in content:
            answer_line = [line for line in content.split('\n') if 'Answer:' in line]
            if answer_line:
                print(f"🎯 提取的答案: {answer_line[-1]}")
        
    except Exception as e:
        print(f"❌ 复杂问题求解失败: {str(e)}")

def test_concurrent_requests():
    """测试并发请求能力"""
    import asyncio
    from openai import AsyncOpenAI
    
    async def single_request(client, model, prompt, request_id):
        try:
            start_time = time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            end_time = time.time()
            return {
                "id": request_id,
                "success": True,
                "time": end_time - start_time,
                "content": response.choices[0].message.content[:50] + "..."
            }
        except Exception as e:
            return {
                "id": request_id,
                "success": False,
                "error": str(e)
            }
    
    async def test_concurrent():
        client = AsyncOpenAI(
            base_url="http://35.220.164.252:3888/v1/",
            api_key="sk-MjWWt8xBcWvlcVRn7VoMAPSck5zOBbmrR2LXpWcBnBttXeRw"
        )
        
        print("\n🚀 测试并发请求能力...")
        
        # 创建3个并发请求
        prompts = [
            "What is 10 + 15?",
            "What is 20 * 3?",
            "What is 100 / 4?"
        ]
        
        start_time = time.time()
        tasks = []
        for i, prompt in enumerate(prompts):
            task = single_request(client, "gemini-2.5-pro", prompt, i+1)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"⏱️ 总耗时: {end_time - start_time:.2f}秒")
        
        success_count = 0
        for result in results:
            if result["success"]:
                print(f"   ✅ 请求{result['id']}: 成功 ({result['time']:.2f}秒) - {result['content']}")
                success_count += 1
            else:
                print(f"   ❌ 请求{result['id']}: 失败 - {result['error']}")
        
        print(f"📊 成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    try:
        asyncio.run(test_concurrent())
    except Exception as e:
        print(f"❌ 并发测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("🧪 开始API测试...")
    print("=" * 60)
    
    # 基本连接测试
    if test_api_connection():
        print("\n" + "=" * 60)
        
        # 多模型测试
        test_multiple_models()
        print("\n" + "=" * 60)
        
        # 复杂数学问题测试
        test_math_problem()
        print("\n" + "=" * 60)
        
        # 并发测试
        test_concurrent_requests()
        
    print("\n" + "=" * 60)
    print("🎉 测试完成!")

if __name__ == "__main__":
    main()